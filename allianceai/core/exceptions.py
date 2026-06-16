"""Domain-specific exceptions for AllianceAI."""


class AllianceAIError(Exception):
    """Base class for all AllianceAI errors."""


class DataFetchError(AllianceAIError):
    """Raised when a data source cannot be reached or returns unusable data."""


class InsufficientDataError(AllianceAIError):
    """Raised when there is too little data to perform the requested analysis."""


class ModelError(AllianceAIError):
    """Raised when a forecasting or ML model fails to fit or predict."""
