# ClearML - Example of manual graphs and statistics reporting
#
from clearml import Task, Logger


def report_scalars(logger):
    """
    reporting scalars to scalars section
    :param logger: The task.logger to use for sending the scalars
    """
    # report two scalar series on the same graph
    for i in range(100):
        logger.report_scalar(title="unified graph", series="series A", iteration=i, value=1./(i+1))
        logger.report_scalar(title="unified graph", series="series B", iteration=i, value=10./(i+1))

    # report two scalar series on two different graphs
    for i in range(100):
        logger.report_scalar(title="graph A", series="series A", iteration=i, value=1./(i+1))
        logger.report_scalar(title="graph B", series="series B", iteration=i, value=10./(i+1))


def main():
    # Connecting ClearML with the current process,
    # from here on everything is logged automatically
    task = Task.init(project_name="HuggingFace Transformers", task_name="Scalar reporting")

    print('reporting scalar graphs')

    # Get the task logger,
    # You can also call Task.current_task().get_logger() from anywhere in your code.
    logger = task.get_logger()

    # report scalars
    report_scalars(logger)

    # force flush reports
    # If flush is not called, reports are flushed in the background every couple of seconds,
    # and at the end of the process execution
    logger.flush()

    print('We are done reporting, have a great day :)')


if __name__ == "__main__":
    main()
