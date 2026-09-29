def format_number(n):
    if abs(n) >= 1000:
        return f"{n:.2e}"
    if abs(n) < 1e-3:
        return f"{n:.2e}"
    return f"{n:.3f}"



class tcadcommand:
    def __init__(self):
        self.commandstr=""
        self.commandname=""
        self.parameters=[]
        self.extra_parameters=[]
    
    def get_command(self):
        self.commandstr=f"{self.commandname}"
        for i in self.parameters:
            if i=="":
                continue
            method=getattr(self,i)
            if method.value is not None and method.value is not True:
                if isinstance(method.value, (int, float)):
                    self.commandstr+=f" {method.name}={format_number(method.value)}".lower()
                else:
                    self.commandstr+=f" {method.name}={method.value}".lower()
            elif method.value is True:
                self.commandstr+=f" {method.name}".lower()
        if hasattr(self,"extra_parameters"):
            for i in self.extra_parameters:
                self.commandstr+=f" {i}".lower()
        return self.commandstr
            


class tcadparameter:
    def __init__(self,name,ttype,tdefault,units):
        self.name=name
        self.ttype=ttype
        self.tdefault=tdefault
        self.units=units
        self.description=''
        self.description_zh=''
        self.value=None