#!/usr/bin/env python3
"""Generate exp2 scripts (S1-GPU1 through S3-GPU1)."""
import os

configs = [
    ("s1_gpu1", 1, "Server1-GPU1", "mlm_ntp",          "--mlm --ntp"),
    ("s2_gpu0", 0, "Server2-GPU0", "mlm_ssp",          "--mlm --ssp"),
    ("s2_gpu1", 1, "Server2-GPU1", "mlm_cpcl",         "--mlm --cpcl"),
    ("s3_gpu0", 0, "Server3-GPU0", "mlm_pair",         "--mlm --pairing"),
    ("s3_gpu1", 1, "Server3-GPU1", "mlm_ntp_cpcl_pair","--mlm --ntp --cpcl --pairing"),
]

template = open(os.path.join(os.path.dirname(__file__), "_exp2_template.sh")).read()
base_dir = os.path.dirname(__file__)

for label, gpu, server_label, config_name, config_flags in configs:
    content = template \
        .replace("__GPU__", str(gpu)) \
        .replace("__CONFIG_NAME__", config_name) \
        .replace("__CONFIG_FLAGS__", config_flags) \
        .replace("__SERVER_LABEL__", server_label)

    filepath = os.path.join(base_dir, f"exp2_{label}.sh")
    with open(filepath, "w") as f:
        f.write(content)
    os.chmod(filepath, 0o755)
    print(f"Created: {filepath}")

print("Done.")
