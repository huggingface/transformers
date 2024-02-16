import sys
import traceback
import inspect


class Provenance:
    def __init__(self) -> None:
        self.initial_stacktrace = self.stacktrace()

    def stacktrace(self):
        trace = inspect.stack()
        output = []

        for frame in trace[2:-1]:
            try:
                frame = f'{frame.filename}:{frame.lineno}:{frame.function}:   {frame.code_context[0].strip()}'
                output.append(frame)
            except:
                output.append(None)

        return output

    def toDict(self):  # for ujson
        self.serialization_stacktrace = self.stacktrace()
        return dict(self.__dict__)


if __name__ == '__main__':
    p = Provenance()
    print(p.toDict().keys())

    import ujson
    print(ujson.dumps(p, indent=4))


    class X:
        def __init__(self) -> None:
            pass
        
        def toDict(self):
            return {'key': 1}
    
    print(ujson.dumps(X()))