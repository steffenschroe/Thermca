import math


def round_to_digits(number, num_digits):
    """Round float to given number of significant digits"""
    return (
        number
        if number == 0
        else round(number, -int(math.floor(math.log10(abs(number)))) + (num_digits - 1))
    )


class TextProgressBar:
    def __init__(self, min=0, max=0, description=''):
        self.min = min
        self.max = max
        self._value = 0
        self._print_count = 0
        self.speed = 0.0
        self.finished = False
        print(description)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v):
        self._value = v
        print(f"{int(v*100./(self.max-self.min)): >2}% ", end='')
        self._print_count += 1
        if (self._print_count * 4) % 72 == 0:
            speed = round_to_digits(self.speed, 2)
            time = round_to_digits(v, 3)
            print(f" time: {time}, speed: {speed}")

    def finish(self):
        self.finished = True
        print('100% done.')
