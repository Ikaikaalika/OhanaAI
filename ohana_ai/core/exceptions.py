"""Custom exceptions for OhanaAI."""


class OhanaAIError(Exception):
    """Base exception for all OhanaAI errors."""

    def __init__(self, message: str, details: str = None) -> None:
        """Initialize OhanaAI error.

        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(message)
        self.message = message
        self.details = details

    def __str__(self) -> str:
        """Return string representation of error."""
        if self.details:
            return f"{self.message}: {self.details}"
        return self.message


class ConfigError(OhanaAIError):
    """Error in configuration loading or validation."""

    pass


class GedcomParseError(OhanaAIError):
    """Error parsing GEDCOM files."""

    def __init__(
        self, message: str, file_path: str = None, line_number: int = None
    ) -> None:
        """Initialize GEDCOM parse error.

        Args:
            message: Error message
            file_path: Path to GEDCOM file with error
            line_number: Line number where error occurred
        """
        super().__init__(message)
        self.file_path = file_path
        self.line_number = line_number

    def __str__(self) -> str:
        """Return string representation of error."""
        parts = [self.message]
        if self.file_path:
            parts.append(f"File: {self.file_path}")
        if self.line_number:
            parts.append(f"Line: {self.line_number}")
        return " | ".join(parts)


class ModelError(OhanaAIError):
    """Error in model training or inference."""

    pass


class ValidationError(OhanaAIError):
    """Error in data validation."""

    pass
