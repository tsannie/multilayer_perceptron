def check_arguments(valid_options):
    def decorator(func):
        def wrapper(self, argument):
            if isinstance(argument, str):
                if argument not in valid_options:
                    raise ValueError(
                        "Invalid argument. {} not in list of valid options.".format(
                            argument
                        )
                    )
                argument = valid_options[argument]
                argument = argument()
            else:
                is_valid = False
                for _, v in valid_options.items():
                    if isinstance(argument, v):
                        is_valid = True
                        break
                if not is_valid:
                    raise TypeError("Invalid argument type.")

            return func(self, argument)

        return wrapper

    return decorator
