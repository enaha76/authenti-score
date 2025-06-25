from io import BytesIO
from PIL import Image

try:
    from open_c2pa import ManifestStore
except Exception:  # pragma: no cover - library may not be installed during tests
    ManifestStore = None


def detect_c2pa(image: Image.Image) -> bool:
    """Detect C2PA manifests mentioning DALL·E or ChatGPT.

    Parameters
    ----------
    image: PIL.Image.Image
        Image to inspect for embedded C2PA metadata.

    Returns
    -------
    bool
        ``True`` if a manifest referencing DALL·E or ChatGPT is found,
        ``False`` otherwise.
    """

    if ManifestStore is None:
        # Library not available; unable to detect
        return False

    try:
        with BytesIO() as buf:
            fmt = image.format or "PNG"
            image.save(buf, format=fmt)
            data = buf.getvalue()

        store = ManifestStore.from_bytes(data)
        if not store or not store.claim:
            return False

        claim = store.claim

        generator = (getattr(claim, "generator", "") or "").lower()
        if "dall" in generator or "chatgpt" in generator:
            return True

        for ingredient in getattr(claim, "ingredients", []):
            title = (getattr(ingredient, "title", "") or "").lower()
            if "dall" in title or "chatgpt" in title:
                return True

    except Exception:
        # Any parsing errors imply no valid watermark detected
        return False

    return False

