import re, math


class Number:
    def __init__(self, v):
        self.v = v


class Variable:
    def __init__(self, n):
        self.n = n


class BinOp:
    def __init__(self, l, op, r):
        self.l = l
        self.op = op
        self.r = r


class UnaryFunc:
    def __init__(self, name, arg):
        self.name = name
        self.arg = arg


class Parser:
    """
    Recursive descent base parser
    """

    TOKEN_RE = re.compile(
        r"\s*(?:(\d+(\.\d*)?)|([A-Za-z_][A-Za-z_0-9]*)|(\*\*|[()+\-*/^~]))"
    )

    def __init__(self, functions=None, operators=None, constants=None):
        # Default unary functions
        self.functions = functions or {
            "abs": abs,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "exp": math.exp,
            "log": math.log,
        }
        # Default binary operators
        self.operators = operators or {
            "+": lambda a, b: a + b,
            "-": lambda a, b: a - b,
            "*": lambda a, b: a * b,
            "/": lambda a, b: a / b,
            "^": lambda a, b: a**b,
            "%": lambda a, b: a % b,
        }
        # Default constants
        self.constants = constants or {
            "pi": math.pi,
            "e": math.e,
        }

    def add_function(self, name, func):
        self.functions[name] = func

    def add_operator(self, symbol, func):
        self.operators[symbol] = func

    def add_constant(self, name, value):
        self.constants[name] = value

    def tokenize(self, expr):
        for number, _, name, op in self.TOKEN_RE.findall(expr):
            if number:
                yield ("NUMBER", float(number))
            elif name:
                yield ("NAME", name)
            else:
                yield ("OP", op)

    def parse(self, expr):
        self.tokens = list(self.tokenize(expr))
        self.pos = 0
        return self.expr()

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else ("EOF", None)

    def consume(self):
        tok = self.peek()
        self.pos += 1
        return tok

    def expr(self):
        node = self.term()
        while self.peek()[1] in ("+", "-"):
            op = self.consume()[1]
            node = BinOp(node, op, self.term())
        return node

    def term(self):
        node = self.factor()
        while self.peek()[1] in ("*", "/"):
            op = self.consume()[1]
            node = BinOp(node, op, self.factor())
        return node

    def factor(self):
        return self.power()

    def power(self):
        node = self.atom()
        if self.peek()[1] == "**":
            self.consume()
            node = BinOp(node, "**", self.power())
        return node

    def atom(self):
        tok_type, tok_val = self.consume()
        if tok_type == "NUMBER":
            return Number(tok_val)
        elif tok_type == "NAME":
            # Function call with parentheses
            if self.peek()[1] == "(":
                self.consume()
                arg = self.expr()
                if self.consume()[1] != ")":
                    raise SyntaxError("Expected ')'")
                return UnaryFunc(tok_val, arg)
            # Unary function
            elif tok_val in self.functions:
                return UnaryFunc(tok_val, self.atom())
            # Variable or constant
            return Variable(tok_val)
        elif tok_val == "(":
            node = self.expr()
            if self.consume()[1] != ")":
                raise SyntaxError("Expected ')'")
            return node
        elif tok_val in self.functions:
            return UnaryFunc(tok_val, self.atom())
        else:
            raise SyntaxError(f"Unexpected token: {tok_val}")

    def eval(self, node, vars={}):
        if isinstance(node, Number):
            return node.v
        if isinstance(node, Variable):
            if node.n in vars:
                return vars[node.n]
            elif node.n in self.constants:
                return self.constants[node.n]
            else:
                raise NameError(f"Unknown variable {node.n}")
        if isinstance(node, BinOp):
            l = self.eval(node.l, vars)
            r = self.eval(node.r, vars)
            if node.op not in self.operators:
                raise NameError(f"Unknown operator {node.op}")
            return self.operators[node.op](l, r)
        if isinstance(node, UnaryFunc):
            if node.name not in self.functions:
                raise NameError(f"Unknown function {node.name}")
            arg_val = self.eval(node.arg, vars)
            return self.functions[node.name](arg_val)

    def compute(self, expr, vars={}):
        tree = self.parse(expr)
        return self.eval(tree, vars)
