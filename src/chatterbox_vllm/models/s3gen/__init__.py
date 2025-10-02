from .monkey_patches import apply_monkey_patches

apply_monkey_patches()


from .s3gen import S3Token2Wav as S3Gen
from .const import S3GEN_SR
