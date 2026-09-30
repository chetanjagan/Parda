"""An export-friendly replacement for GLiNER's packed LSTM encoder (modeling/layers.py).

GLiNER runs a bidirectional LSTM over the words with pack_padded_sequence. During ONNX export the lengths it packs
with come from a plain Python list GLiNER passes in (word_lengths), so the exporter freezes them into the model and
any text of another length crashes ("Invalid value/s in sequence_lens"). This module computes the SAME result without
packing, from lengths given as a real input:

  forward direction   the LSTM over the padded sequence (padding sits at the end, so real positions are unaffected)
  backward direction  each sequence is reversed within its own length, run forward, and reversed back
  after the length    zeros, exactly like pad_packed_sequence
"""
import inspect
import types


def _uni(lstm, layer, reverse):
    """A one-layer, one-direction nn.LSTM holding the weights of `lstm` for (layer, direction)."""
    import torch
    sfx = f"_l{layer}" + ("_reverse" if reverse else "")
    in_size = lstm.input_size if layer == 0 else lstm.hidden_size * (2 if lstm.bidirectional else 1)
    uni = torch.nn.LSTM(in_size, lstm.hidden_size, num_layers=1, bias=lstm.bias, batch_first=True)
    with torch.no_grad():
        for name in ("weight_ih", "weight_hh") + (("bias_ih", "bias_hh") if lstm.bias else ()):
            getattr(uni, f"{name}_l0").copy_(getattr(lstm, f"{name}{sfx}"))
    for p in uni.parameters():  # fixed copies: the exporter can only bake in weights that are not trainable
        p.requires_grad_(False)
    return uni.eval()


def masked_lstm(lstm, parts, x, lengths):
    """pack_padded_sequence + lstm + pad_packed_sequence(batch_first=True), without packing. x: (B, T, D)."""
    import torch
    T = x.shape[1]
    t = torch.arange(T, device=x.device).unsqueeze(0)
    lengths = lengths.to(x.device).view(-1, 1)
    valid = t < lengths
    rev = torch.where(valid, lengths - 1 - t, t)  # reverse inside each length; applying it twice gives t back
    inp = x
    for layer in range(lstm.num_layers):
        outs = []
        for d, uni in enumerate(parts[layer]):
            seq = inp if d == 0 else torch.gather(inp, 1, rev.unsqueeze(-1).expand(-1, -1, inp.shape[-1]))
            h, _ = uni(seq)
            if d == 1:
                h = torch.gather(h, 1, rev.unsqueeze(-1).expand(-1, -1, h.shape[-1]))
            outs.append(h)
        inp = torch.cat(outs, -1) * valid.unsqueeze(-1).to(x.dtype)
    return inp


def _packs(mod):
    try:
        return "pack_padded_sequence" in inspect.getsource(type(mod).forward)
    except (OSError, TypeError):
        return False


def find_packed_lstms(net):
    """[(name, module, lstm attribute)] for sub-modules (not the network itself) that pack their input into
    exactly one nn.LSTM child, whatever that child is called."""
    import torch
    found = []
    for name, mod in net.named_modules():
        if mod is net or not _packs(mod):
            continue
        lstms = [a for a, c in mod.named_children() if isinstance(c, torch.nn.LSTM)]
        if len(lstms) == 1:
            found.append((name, mod, lstms[0]))
    return found


def packing_modules(net):
    """Every module (network included) whose forward packs sequences: export problems if left unpatched."""
    return [name or "<network>" for name, mod in net.named_modules() if _packs(mod)]


def patch_lstms(net, holder):
    """Swap the forward of every packed-LSTM encoder in `net`. Lengths come from holder["lengths"] (set by the
    export wrapper from a real input) or, outside export, from the mask exactly as GLiNER computes them."""
    patched = []
    for name, mod, attr in find_packed_lstms(net):
        lstm = getattr(mod, attr)
        mod._parda_lstm = lstm
        if not lstm.batch_first or getattr(lstm, "proj_size", 0):
            raise RuntimeError(f"{name}: only batch_first LSTMs without projection are supported")
        parts = [[_uni(lstm, layer, r) for r in ((False, True) if lstm.bidirectional else (False,))]
                 for layer in range(lstm.num_layers)]
        mod._parda_parts = parts  # plain attribute: not registered as sub-modules

        def forward(self, *args, **kwargs):
            x = args[0] if args else kwargs["x"]
            mask = args[1] if len(args) > 1 else kwargs.get("mask")
            lengths = holder.get("lengths")
            if lengths is None:
                lengths = mask[:, :x.shape[1]].sum(dim=1)
            return masked_lstm(self._parda_lstm, self._parda_parts, x, lengths)

        mod.forward = types.MethodType(forward, mod)
        patched.append(name)
    return patched
