"""
Process Reward Model (PRM) for scoring individual reasoning steps.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import List, Tuple, Optional, Dict
import numpy as np
from tqdm import tqdm

from ..config import PRMConfig, TrainingConfig


class ProcessRewardModel(nn.Module):
    """
    Process Reward Model that scores each step of a reasoning trace.

    Architecture:
        Base LM encoder -> step-level pooling -> classification head
    """

    def __init__(self, config: PRMConfig):
        super().__init__()
        self.config = config

        # Auto-detect hidden size from model config if not specified
        if config.hidden_size is None:
            hf_config = AutoConfig.from_pretrained(
                config.reward_model_name, trust_remote_code=True
            )
            self.hidden_size = hf_config.hidden_size
        else:
            self.hidden_size = config.hidden_size

        assert config.num_reward_classes >= 2, (
            "num_reward_classes must be >= 2 for score_steps to work"
        )

        self.backbone = AutoModel.from_pretrained(
            config.reward_model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        # Match reward head dtype to backbone
        self.reward_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.hidden_size // 2, config.num_reward_classes),
        ).to(torch.float16)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        step_boundaries: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            step_boundaries: (batch, max_steps) - token positions marking end of each step
            labels: (batch, max_steps) - per-step labels (0=bad, 1=good)
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden)

        if step_boundaries is not None:
            step_rewards = self._extract_step_rewards(hidden_states, step_boundaries)
        else:
            # Use last non-pad token representation (works with right-padding)
            seq_lengths = attention_mask.sum(dim=1) - 1
            step_rewards = self.reward_head(
                hidden_states[torch.arange(hidden_states.size(0)), seq_lengths]
            )
            step_rewards = step_rewards.unsqueeze(1)

        result = {"step_rewards": step_rewards}

        if labels is not None:
            # Flatten for loss computation, ignoring padding (-100)
            valid_mask = labels != -100
            valid_rewards = step_rewards[valid_mask]
            valid_labels = labels[valid_mask]
            if valid_labels.numel() > 0:
                result["loss"] = self.loss_fn(
                    valid_rewards.float(), valid_labels
                )
            else:
                result["loss"] = torch.tensor(0.0, device=input_ids.device)

        return result

    def _extract_step_rewards(
        self, hidden_states: torch.Tensor, step_boundaries: torch.Tensor
    ) -> torch.Tensor:
        """Extract reward scores at each step boundary position."""
        batch_size, max_steps = step_boundaries.shape
        seq_len = hidden_states.size(1)
        step_hidden = []

        for b in range(batch_size):
            for s in range(max_steps):
                pos = step_boundaries[b, s].item()
                if pos >= 0:
                    # Clamp to valid range to handle truncation
                    pos = min(pos, seq_len - 1)
                    step_hidden.append(hidden_states[b, pos])
                else:
                    step_hidden.append(torch.zeros_like(hidden_states[b, 0]))

        step_hidden = torch.stack(step_hidden).view(batch_size, max_steps, -1)
        return self.reward_head(step_hidden)

    def score_steps(
        self,
        text: str,
        tokenizer: AutoTokenizer,
        step_separator: str = "\n\n",
        device: str = None,
    ) -> List[float]:
        """Score each reasoning step in a text."""
        # Infer device from model parameters
        if device is None:
            device = next(self.parameters()).device
        else:
            device = torch.device(device)

        steps = [s.strip() for s in text.split(step_separator) if s.strip()]
        scores = []

        # Score each step by encoding the trace up to that step
        for i in range(len(steps)):
            partial = step_separator.join(steps[: i + 1])
            inputs = tokenizer(
                partial, return_tensors="pt", truncation=True, max_length=1024
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.forward(**inputs)
                reward_logits = outputs["step_rewards"][:, -1, :]
                prob_good = torch.softmax(reward_logits.float(), dim=-1)[0, 1].item()
                scores.append(prob_good)

        return scores

    def aggregate_score(self, step_scores: List[float], method: str = "min") -> float:
        """Aggregate step-level scores into a single trace score."""
        if not step_scores:
            return 0.0
        if method == "min":
            return min(step_scores)
        elif method == "mean":
            return float(np.mean(step_scores))
        elif method == "last":
            return step_scores[-1]
        elif method == "product":
            return float(np.prod(step_scores))
        else:
            raise ValueError(f"Unknown aggregation method: {method}")


class StepLabelDataset(Dataset):
    """Dataset for PRM training with step-level labels."""

    def __init__(
        self,
        traces: List[Dict],
        tokenizer: AutoTokenizer,
        max_length: int = 1024,
        step_separator: str = "\n\n",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.step_separator = step_separator
        self.pad_token_id = tokenizer.pad_token_id or 0
        self.examples = self._prepare_examples(traces)

    def _prepare_examples(self, traces: List[Dict]) -> List[Dict]:
        """Convert traces with step labels into tokenized examples."""
        examples = []

        for trace in traces:
            steps = trace["steps"]
            labels = trace["step_labels"]  # list of 0/1 per step

            full_text = self.step_separator.join(steps)
            encoding = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            # Find step boundary positions in token space
            boundaries = []
            current_text = ""
            for step in steps:
                if current_text:
                    current_text += self.step_separator
                current_text += step
                step_tokens = self.tokenizer(
                    current_text,
                    truncation=True,
                    max_length=self.max_length,
                )
                boundary_pos = len(step_tokens["input_ids"]) - 1
                # Only keep boundaries within the truncated sequence
                if boundary_pos < self.max_length:
                    boundaries.append(boundary_pos)
                else:
                    boundaries.append(self.max_length - 1)

            examples.append({
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "step_boundaries": torch.tensor(boundaries),
                "labels": torch.tensor(labels),
            })

        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class PRMTrainer:
    """Trainer for the Process Reward Model."""

    def __init__(
        self,
        model: ProcessRewardModel,
        train_dataset: StepLabelDataset,
        eval_dataset: Optional[StepLabelDataset] = None,
        config: Optional[TrainingConfig] = None,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config if config is not None else TrainingConfig()
        self.pad_token_id = getattr(train_dataset, 'pad_token_id', 0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Set seed for reproducibility
        torch.manual_seed(self.config.seed)

    def train(self):
        """Full training loop."""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
        )

        num_steps = len(train_loader) * self.config.num_epochs
        warmup_steps = int(num_steps * self.config.warmup_ratio)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, num_steps - warmup_steps
        )

        # FP16 mixed precision
        scaler = None
        if self.config.fp16 and self.device.type == "cuda":
            scaler = torch.amp.GradScaler("cuda")

        self.model.train()
        global_step = 0

        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.config.num_epochs}")

            for batch_idx, batch in enumerate(pbar):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                if scaler:
                    with torch.amp.autocast("cuda"):
                        outputs = self.model(**batch)
                        loss = outputs["loss"] / self.config.gradient_accumulation_steps
                    scaler.scale(loss).backward()
                else:
                    outputs = self.model(**batch)
                    loss = outputs["loss"] / self.config.gradient_accumulation_steps
                    loss.backward()

                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    if scaler:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.max_grad_norm
                        )
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.max_grad_norm
                        )
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                epoch_loss += loss.item() * self.config.gradient_accumulation_steps
                pbar.set_postfix({"loss": f"{epoch_loss / (batch_idx + 1):.4f}"})

                if global_step > 0 and global_step % self.config.eval_steps == 0 and self.eval_dataset:
                    eval_loss = self.evaluate()
                    print(f"  Step {global_step} | Eval Loss: {eval_loss:.4f}")
                    self.model.train()

            # Flush remaining gradients from incomplete accumulation window
            if (batch_idx + 1) % self.config.gradient_accumulation_steps != 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        return self.model

    def evaluate(self):
        """Evaluate the model on eval dataset."""
        if self.eval_dataset is None:
            return 0.0

        self.model.eval()
        eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.config.batch_size,
            collate_fn=self._collate_fn,
        )

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                if "loss" in outputs:
                    total_loss += outputs["loss"].item()
                num_batches += 1

        return total_loss / max(num_batches, 1)

    def _collate_fn(self, batch):
        """Pad and collate a batch of examples."""
        max_len = max(ex["input_ids"].size(0) for ex in batch)
        max_steps = max(ex["step_boundaries"].size(0) for ex in batch)

        input_ids = []
        attention_masks = []
        boundaries = []
        labels = []

        for ex in batch:
            seq_len = ex["input_ids"].size(0)
            num_steps = ex["step_boundaries"].size(0)

            # Pad sequences with proper pad token id
            pad_len = max_len - seq_len
            input_ids.append(
                torch.cat([ex["input_ids"], torch.full((pad_len,), self.pad_token_id, dtype=torch.long)])
            )
            attention_masks.append(
                torch.cat([ex["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
            )

            # Pad step boundaries and labels
            step_pad = max_steps - num_steps
            boundaries.append(
                torch.cat([ex["step_boundaries"], torch.full((step_pad,), -1, dtype=torch.long)])
            )
            labels.append(
                torch.cat([ex["labels"], torch.full((step_pad,), -100, dtype=torch.long)])
            )

        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_masks),
            "step_boundaries": torch.stack(boundaries),
            "labels": torch.stack(labels),
        }
