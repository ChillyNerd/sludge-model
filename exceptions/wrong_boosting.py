class WrongBoostingType(Exception):
    def __init__(self):
        super().__init__("Wrong boosting type!")
