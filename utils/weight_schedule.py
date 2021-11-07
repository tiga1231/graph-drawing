import bisect

##private
class LinearStepSchedule:
    def __init__(self, x0=0, x1=1.5e5, y0=1, y1=0.01):
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        
    def __call__(self, x):
        x0,x1,y0,y1 = self.x0,self.x1,self.y0,self.y1
        if x <= x0:
            return y0
        if x0 < x <= x1:
            return (y1-y0)/(x1-x0) * (x-x0) + y0
        else:
            return y1

##private
class SmoothStepSchedule:
    ## y=3x^2-2x^3
    def __init__(self, x0=0, x1=1.5e5, y0=1, y1=0.01):
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        
    def __call__(self, x):
        x0,x1,y0,y1 = self.x0,self.x1,self.y0,self.y1
        if x <= x0:
            return y0
        if x0 < x <= x1:
            x = (x-x0) / (x1-x0)
            y = 3*x**2 - 2*x**3
            y = y0 + y*(y1-y0)
            return y
        else:
            return y1
        
        
##private
class Concat:
    def __init__(self, *schedules):
        '''
        Assuming s1.x0 < s1.x1 < s2.x0 < s2.x1 < ...
        for schedules = [s1, s2, ...]
        '''
        self.schedules = schedules
#         self.xs = sum([[s.x0, s.x1] for s in schedules],[])
#         self.ys = sum([[s.y0, s.y1] for s in schedules],[])
        self.switches = [(s1.x1+s2.x0)/2 for s1,s2 in zip(schedules[:-1], schedules[1:])] 
    def __call__(self, x):
        i = bisect.bisect_left(self.switches, x)
        return self.schedules[i](x)
    
    
##public
class SmoothSteps:
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys
        self.schedule = Concat(*[
            SmoothStepSchedule(x0,x1,y0,y1) 
            for x0,x1,y0,y1 in zip(xs[:-1], xs[1:], ys[:-1], ys[1:])
        ])
        
    def __call__(self, x):
        return self.schedule(x)
    
##public
class LinearSteps:
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys
        self.schedule = Concat(*[
            LinearStepSchedule(x0,x1,y0,y1) 
            for x0,x1,y0,y1 in zip(xs[:-1], xs[1:], ys[:-1], ys[1:])
        ])
        
    def __call__(self, x):
        return self.schedule(x)