class Solver():
    def __init__(self, board, priority):
        #Инициализация работы класса
        self.board = board #инициализация доски
        self.priority = priority #инициализация приоритета
        self.solution = None
        self.solution_moves = -1 #инициализация количества шагов требуемых для решения
        self.solved = False #инициализация решаемости состояния
        self.search_nodes = 0

    def isSolvable(self): #проверяет имеет ли начальное состояние решение
        return self.solved

    def moves(self): # возвращает количество шагов если решение имеется
        return self.solution_moves if self.solved else -1

    def solve(self): # основной метод реализации алгортим A*
        pass

    def __iter__(self):
        pass

    def __next__(self):
        pass

