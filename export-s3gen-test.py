import torch
from chatterbox_vllm.models.s3gen import S3Gen
from chatterbox_vllm.models.s3gen.s3gen_exportable import S3GenExportable

scan_for_errors = False

# Load original model
base = S3Gen(use_fp16=False).eval().cuda()

# Wrap
exportable = S3GenExportable(base.flow, base.mel2wav, n_timesteps=10).eval().cuda()

def script_or_report(name, mod):
    try:
        torch.jit.script(mod)
        return True
    except Exception as e:
        print(f"[JIT FAIL] {name} :: {type(mod)}\n  -> {e}\n")
        return False

def find_bad_leaf(root, prefix=""):
    # Check direct children first
    for n, m in root.named_children():
        full = f"{prefix}.{n}" if prefix else n
        if not script_or_report(full, m):
            # Dive deeper only into failing subtrees
            find_bad_leaf(m, full)

if scan_for_errors:
    find_bad_leaf(exportable)
else:
    # Dummy inputs
    B = 1
    speech_tokens = torch.randint(0, 6500, (B, 100)).cuda()
    prompt_token = torch.randint(0, 6500, (B, 50)).cuda()
    prompt_token_len = torch.tensor([50], device="cuda")
    prompt_feat = torch.randn(B, 100, 80).cuda()
    prompt_feat_len = torch.tensor([100], device="cuda")
    embedding = torch.randn(B, 80).cuda()

    example_mu = torch.randn(1, 80, 100)
    example_mask = torch.ones(1, 1, 100)
    scripted = torch.jit.trace(exportable, (speech_tokens, prompt_token, prompt_token_len, prompt_feat, prompt_feat_len, embedding))
    scripted.save("s3gen_scripted.pt")

    # TorchScript export
    #scripted = torch.jit.script(exportable)
    #scripted.save("s3gen_scripted.pt")

    # Test reload
    loaded = torch.jit.load("s3gen_scripted.pt").cuda()
    wav = loaded(speech_tokens, prompt_token, prompt_token_len,
                prompt_feat, prompt_feat_len, embedding)
    print(wav.shape)