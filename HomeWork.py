from queue import PriorityQueue


class Solver():
    def __init__(self, board, priority):
        #Инициализация работы класса
        self.board = board #инициализация доски
        self.priority = priority #инициализация приоритета
        self.solution = None # инициализация списка, содержащего последовательность состояний доски
        self.solution_moves = -1 #инициализация количества шагов требуемых для решения
        self.solved = False #инициализация решаемости состояния
        self.search_nodes = 0 #инициализация счётчика состояний игры
        self.visited = {} # инициализация словаря для хранения состояний
        self.solve() # вызов метода реализованного агоритмом A*

    def isSolvable(self): #проверяет имеет ли начальное состояние решение
        return self.solved

    def moves(self): # возвращает количество шагов если решение имеется
        return self.solution_moves if self.solved else -1

    def solve(self): # основной метод реализации алгортим A*
        if self.board.is_goal(): #проверка на начальное состояние
            self.solved = True #решение есть
            self.solution_moves = 0 #количество шагов для решения 0
            self.solution = [self.board] #состояние доски окончательное
            return

    def __iter__(self):
        if self.solution:
            return iter(self.solution)
        return iter([])

    def __next__(self):
        if not hasattr(self, '_iter'):
            self._iter = iter(self.solution) if self.solution else iter([])
        return next(self._iter)
