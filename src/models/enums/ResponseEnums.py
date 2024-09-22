from enum import Enum

class ResponseSignal(Enum):

    FILE_VALIDATED_SUCCESS = "file_validate_successfully"
    FILE_TYPE_NOT_SUPPORTED = "file_type_not_supported"
    FILE_SIZE_EXCEEDED = "file_size_exceeded"
    FILE_UPLOAD_SUCCESS = "file_upload_success"
    FILE_UPLOAD_FAILED = "file_upload_failed"
    FILE_NOT_FOUND = "file_not_found"
    FILE_FOUND = "file_found"
    UnicodeDecodeError = "Error decoding the file, check the encoding."
    SUCCESS = "success"
    
