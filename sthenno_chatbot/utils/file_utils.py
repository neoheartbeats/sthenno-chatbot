import mimetypes

mimetypes.init()


def image_p(url: str) -> bool:
    if mimetypes.guess_type(url)[0] is not None:
        if "image" in mimetypes.guess_type(url)[0]:  # type: ignore
            return True
    return False
