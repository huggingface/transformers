# Example usage of the trace and attach_tracer decorators

from transformers.utils.metrics import attach_tracer, traced


@attach_tracer()
class ExampleClass:
    def __init__(self, name):
        # The attach_tracer decorator has already created self.tracer for us
        self.name = name

    @traced  # This method will use the tracer from the class instance
    def process_data(self, data):
        # This method is traced and can use self.tracer
        return f"Processed {data} with {self.name}"

    @traced(span_name="custom_operation")  # With custom span name
    def special_operation(self, value):
        # Also traced, with a custom span name
        return value * 2

    @traced(
        additional_attributes=[
            ("name", "object.name", lambda x: x.upper()),  # Using a transform function
            ("name", "object.fixed_value", "static_value"),  # Using a fixed value
        ]
    )
    def operation_with_attributes(self):
        # This will add the specified attributes to the span
        return "Operation completed"


# For functions without a class, the traced decorator still works
@traced
def standalone_function(arg1, arg2):
    # For functions, a tracer is created based on the module name
    return arg1 + arg2


# Usage:
if __name__ == "__main__":
    # With OpenTelemetry configured, these will produce traces
    example = ExampleClass("test_object")
    example.process_data("sample")
    example.special_operation(42)
    example.operation_with_attributes()

    result = standalone_function(1, 2)
