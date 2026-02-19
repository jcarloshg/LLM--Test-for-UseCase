class CustomResponse:
    def __init__(self, msg: str = "", is_success: bool = True, data: None = None):
        self.message = msg
        self.is_success = is_success
        self.data = data

    @staticmethod
    def success(msg: str = "Success", success: bool = True, data: None = None):
        return CustomResponse(
            msg=msg,
            is_success=success,
            data=data
        )

    @staticmethod
    def error(msg: str = "Error"):
        return CustomResponse(
            msg=msg,
            is_success=False,
            data=None
        )
