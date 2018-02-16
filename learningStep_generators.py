def harmonicSeries(start):
    i=0
    while True:
        yield start/(1+i*0.1)
        i+=1
        
def constant(step):
    while True:
        yield step