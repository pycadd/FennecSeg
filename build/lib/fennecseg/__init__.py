"""fennecseg: fast 3D CT organ segmentation with self-describing model packages."""
__version__ = "0.2.0"

from .inference import predict  # noqa: E402
from .io import read_image  # noqa: E402
from .model import load  # noqa: E402

__all__ = ["load", "predict", "read_image", "__version__"]
