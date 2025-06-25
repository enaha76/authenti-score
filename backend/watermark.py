from io import BytesIO
from typing import Optional, Tuple
from PIL import Image

try:
    from open_c2pa import ManifestStore
except Exception:  # pragma: no cover - library may not be installed during tests
    ManifestStore = None


def detect_c2pa(image: Image.Image) -> Tuple[bool, Optional[str]]:
    """Detect any embedded C2PA watermark information.

    Parameters
    ----------
    image: PIL.Image.Image
        Image to inspect for embedded C2PA metadata.

    Returns
    -------
    Tuple[bool, Optional[str]]
        ``True`` if C2PA metadata is found. The second element contains
        the detected generator name if available.
    """

    if ManifestStore is None:
        # Library not available; unable to detect
        return False, None

    try:
        with BytesIO() as buf:
            fmt = image.format or "PNG"
            image.save(buf, format=fmt)
            data = buf.getvalue()

        store = ManifestStore.from_bytes(data)
        if not store or not store.claim:
            return False, None

        claim = store.claim

        generator = (getattr(claim, "generator", "") or "").strip()
        if generator:
            return True, generator

        for ingredient in getattr(claim, "ingredients", []):
            gen = (getattr(ingredient, "generator", "") or "").strip()
            if gen:
                return True, gen
            title = (getattr(ingredient, "title", "") or "").strip()
            if title:
                return True, title

    except Exception:
        # Any parsing errors imply no valid watermark detected
        return False, None

    return False, None

