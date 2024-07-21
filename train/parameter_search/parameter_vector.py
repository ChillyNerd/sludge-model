from typing import Dict, List


class ParameterVector:
    def __init__(self, parameters: Dict[str, List]):
        self.parameters = parameters
        self.index_capabilities = {parameter: len(value) for parameter, value in self.parameters.items()}
        self.current_index = {}
        self.stop_iteration = False

    def set_defaults(self):
        self.stop_iteration = False
        for parameter in self.index_capabilities.keys():
            self.current_index[parameter] = 0

    def get_next(self):
        for parameter in self.current_index.keys():
            self.current_index[parameter] += 1
            if self.index_capabilities[parameter] > self.current_index[parameter]:
                return self.get_current_value()
            else:
                self.current_index[parameter] = 0
        return None

    def get_current_value(self):
        result = {}
        for parameter, values in self.parameters.items():
            result[parameter] = values[self.current_index[parameter]]
        return result

    def __iter__(self):
        self.set_defaults()
        return self

    def __next__(self):
        if self.stop_iteration:
            raise StopIteration
        current_value = self.get_current_value()
        self.iterate()
        return current_value

    def iterate(self):
        for parameter in self.current_index.keys():
            self.current_index[parameter] += 1
            if self.index_capabilities[parameter] > self.current_index[parameter]:
                return
            self.current_index[parameter] = 0
        self.stop_iteration = True
