"""Batching, padding, and related utilities.

Anything which has a tensor member should inherit from nn.Module, run the
superclass constructor, and register the tensor as a buffer. This enables the
Trainer to move them to the appropriate device."""

from typing import Callable, List, Optional

import torch
from torch import nn


class PaddedTensor(nn.Module):
    """A tensor and its mask.

    This is ordinarily used for padding a tensor list, so it represents
    one of (source, target, features) for a batch."""

    padded: torch.Tensor
    mask: torch.Tensor

    def __init__(
        self,
        tensorlist: List[torch.Tensor],
        pad_idx: int,
        length_msg_callback: Optional[Callable[[int], None]] = None,
        pad_len: Optional[int] = None,
    ):
        """Constructs the padded tensor from a list of tensors.

        The optional pad_len argument can be used, e.g., to keep all batches
        the exact same length, which improves performance on certain
        accelerators. If not specified, it will be computed using the length
        of the longest input tensor.

        Args:
            tensorlist (List[torch.Tensor]): a list of tensors.
            pad_idx (int): padding index.
            length_msg_callback (Callable[[int], None]): callback for catching
                a violating tensor length.
            pad_len (int, optional): desired length for padding.

        """
        super().__init__()
        if pad_len is None:
            pad_len = max(len(tensor) for tensor in tensorlist)
        if length_msg_callback is not None:
            length_msg_callback(pad_len)
        self.register_buffer(
            "padded",
            torch.stack(
                [
                    self.pad_tensor(tensor, pad_idx, pad_len)
                    for tensor in tensorlist
                ],
            ),
        )
        self.register_buffer("mask", self.padded == pad_idx)

    @staticmethod
    def pad_tensor(
        tensor: torch.Tensor, pad_idx: int, pad_max: int
    ) -> torch.Tensor:
        """Pads a tensor.

        Args:
            tensor (torch.Tensor).
            pad_idx (int): padding index.
            pad_max (int): desired tensor length.

        Returns:
            torch.Tensor.
        """
        padding = pad_max - len(tensor)
        return nn.functional.pad(tensor, (0, padding), "constant", pad_idx)

    def __len__(self) -> int:
        return len(self.padded)

    def lengths(self) -> torch.Tensor:
        """Computes the lengths of all the strings in the tensor.

        By convention we seem to want this tensor on CPU.

        Returns:
            torch.Tensor.
        """
        return (self.mask == 0).sum(dim=1).cpu()


class DecoderOnlyPaddedTensor(PaddedTensor):
    """Padded tensor for decoder only model.
    
    Tracks the sequence, mask (for padding, etc), and the prefix lengths. 
    Prefix lengths are here to track the typically encoder part of the input
    for using this as a prefix LM to solve encoder-decoder tasks.
    """
    padded: torch.Tensor
    mask: torch.Tensor
    prefix_lengths: torch.Tensor
    is_masked_sequence: Optional[bool]
    is_target: Optional[bool]

    def __init__(
        self,
        source_tensorlist: List[torch.Tensor],
        target_tensorlist: Optional[List[torch.Tensor]],
        pad_idx: int,
        is_masked_sequence: Optional[bool]=False,
        is_target: Optional[bool] = False,
        length_msg_callback: Optional[Callable[[int], None]] = None,
        pad_len: Optional[int] = None,
    ):
        super(PaddedTensor, self).__init__()
        self.is_masked_sequence = is_masked_sequence
        self.is_target = is_target
        # TODO: Ensure we do not have EOS on the source tensors already
        # TODO: Clean this up---make separate methods.
        if target_tensorlist is not None:
            # Target for validation; contains JUST the actual target sequence.
            if self.is_target:
                if pad_len is None:
                    pad_len = max(len(t) for t in target_tensorlist)
                if length_msg_callback is not None:
                    length_msg_callback(pad_len)
                tensors = torch.stack(
                    [
                        self.pad_tensor(t, pad_idx, pad_len)
                        for t in target_tensorlist
                    ],
                )
            else:
                if pad_len is None:
                    pad_len = max(
                        len(t1) + len(t2)
                        for t1, t2 in zip(source_tensorlist, target_tensorlist)
                    )
                if length_msg_callback is not None:
                    length_msg_callback(pad_len)
                # During training, target tensors look 
                # the same as the source tensor, but we replace the source info
                # with PADs for ignoring loss on them.
                if self.is_masked_sequence:
                    tensors = torch.stack(
                        [
                            self.pad_concat_tensors(t1, t2, pad_idx, pad_len, pad_indices = t1)
                            for t1, t2 in zip(source_tensorlist, target_tensorlist)
                        ],
                    )
                else:
                    tensors = torch.stack(
                        [
                            self.pad_concat_tensors(t1, t2, pad_idx, pad_len)
                            for t1, t2 in zip(source_tensorlist, target_tensorlist)
                        ],
                    )
        else:
            if pad_len is None:
                pad_len = max(len(t) for t in source_tensorlist)
            if length_msg_callback is not None:
                length_msg_callback(pad_len)
            tensors = torch.stack(
                [
                    self.pad_tensor_left(t, pad_idx, pad_len)
                    for t in source_tensorlist
                ],
            )
        
        # FIXME: We need to account for the left-padding we now have.
        #       Each batch should have a prefix length of num_pads + prefix_length.
        # pad_lens = torch.sum(tensors == pad_idx, dim=1)
        if self.is_target:
            self.prefix_lengths = [0 for _ in target_tensorlist]
        else:
            self.prefix_lengths = [s.size(0) for s in source_tensorlist]
        self.register_buffer("padded", tensors)
        self.register_buffer("mask", tensors == pad_idx)

    @staticmethod
    def pad_tensor_left(
        tensor: torch.Tensor,
        pad_idx: int,
        pad_max: int,
        pad_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pads a tensor.

        Args:
            source_tensor (torch.Tensor).
            target_tensor (torch.Tensor).
            pad_idx (int): padding index.
            pad_max (int): desired tensor length.

        Returns:
            torch.Tensor.
        """
        padding = pad_max - len(tensor)
        padded = nn.functional.pad(tensor, (padding, 0), "constant", pad_idx)
        if pad_indices is not None:
            padded[:padding + len(pad_indices)] = pad_idx

        return padded
    
    @staticmethod
    def pad_concat_tensors(
        source_tensor: torch.Tensor,
        target_tensor: torch.Tensor,
        pad_idx: int,
        pad_max: int,
        pad_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pads a tensor.

        Args:
            source_tensor (torch.Tensor).
            target_tensor (torch.Tensor).
            pad_idx (int): padding index.
            pad_max (int): desired tensor length.

        Returns:
            torch.Tensor.
        """
        tensor = torch.cat((source_tensor, target_tensor))
        padding = pad_max - len(tensor)
        padded = nn.functional.pad(tensor, (padding, 0), "constant", pad_idx)
        if pad_indices is not None:
            padded[:padding + len(pad_indices)] = pad_idx

        return padded


class PaddedBatch(nn.Module):
    """Padded source tensor, with optional padded features and target tensors.

    This represents a padded batch. It is produced by the collator and fed to
    the trainer."""

    source: PaddedTensor
    features: Optional[PaddedTensor]
    target: Optional[PaddedTensor]

    def __init__(self, source, features=None, target=None):
        super().__init__()
        self.register_module("source", source)
        self.register_module("target", target)
        self.register_module("features", features)

    @property
    def has_features(self):
        return self.features is not None

    @property
    def has_target(self):
        return self.target is not None

    def __len__(self) -> int:
        return len(self.source)


class DecoderOnlyPaddedBatch(PaddedBatch):
    """Padded batch for decoder-only models.

    This represents a padded batch. It is produced by the collator and fed to
    the trainer. We expect both a source and target tensor, which are then concatenated
    together, and the source tensor is tracked as the 'prefix' to an LM."""

    sequence: DecoderOnlyPaddedTensor
    masked_sequence: DecoderOnlyPaddedTensor
    # For greedy decoding + eval.
    # FIXME: Do this more elegantly...
    source: DecoderOnlyPaddedTensor
    target: DecoderOnlyPaddedTensor

    def __init__(self, sequence, masked_sequence, source, target):
        super(PaddedBatch, self).__init__()
        self.register_module("sequence", sequence)
        self.register_module("masked_sequence", masked_sequence)
        self.register_module("source", source)
        self.register_module("target", target)

    def __len__(self) -> int:
        return len(self.sequence)