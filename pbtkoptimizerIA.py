from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter.ttk import *
import csv
import tkinter
import numpy as np
from lmfit import minimize, Parameters, Parameter, report_fit
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.figure as Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from lmfit.printfuncs import *
 
window = Tk()
 
window.title("PBTK Tool")
 
tab_control = ttk.Notebook(window)
 
moddef = ttk.Frame(tab_control)
paramdef = ttk.Frame(tab_control)
validdef = ttk.Frame(tab_control)
paramopt = ttk.Frame(tab_control)

tab_control.add(paramdef, text='Parameter Definition') 
tab_control.add(moddef, text='Model Definition')
tab_control.add(validdef, text='Validation Results')
tab_control.add(paramopt, text='Parameter Optimization')

p={}    #Parameters Dictionary
c={}    #Compartments Dictionary
ODEtext={}  #Model Definition Dictionary
validboxes=[] #list of checkboxes for validation

#Parameter Definition
params = Parameters()
def paramsbtnclick():
    paramfilename=filedialog.askopenfilename(filetypes = (("CSV","*.csv"),("all files","*.*")))
    with open(paramfilename, newline='') as paramfile:
        paramreader = csv.reader(paramfile)
        next(paramreader, None)  # skip the headers
        for row in paramreader:
            params.add(row[0])
            if (row[1]):
                params[row[0]].set(value=float(row[1]))
            if (row[2]):
                params[row[0]].set(min=float(row[2]))
            if (row[3]):
                params[row[0]].set(max=float(row[3]))
            if (row[4]):
                params[row[0]].set(max=float(row[4]))
            if (row[5]=='FALSE'):
                params[row[0]].set(vary=FALSE)
            if (row[6]):
                params[row[0]].set(expr=row[6])
            if (row[7]):
                params[row[0]].set(brute_step=float(row[7]))
    Label(paramdef, text='Name').grid(column=0, row=1)
    Label(paramdef, text='Value').grid(column=1, row=1)
    Label(paramdef, text='Min').grid(column=2, row=1)
    Label(paramdef, text='Max').grid(column=3, row=1)
    Label(paramdef, text='Stderr').grid(column=4, row=1)
    Label(paramdef, text='Vary').grid(column=5, row=1)
    Label(paramdef, text='Expr').grid(column=6, row=1)
    Label(paramdef, text='Brute_Step').grid(column=7, row=1)
    i=2
    for par in params:
        Label(paramdef, text=par).grid(column=0, row=i)
        Label(paramdef, text=params[par].value).grid(column=1, row=i)
        Label(paramdef, text=params[par].min).grid(column=2, row=i)
        Label(paramdef, text=params[par].max).grid(column=3, row=i)
        Label(paramdef, text=params[par].stderr).grid(column=4, row=i)
        Label(paramdef, text=params[par].vary).grid(column=5, row=i)
        Label(paramdef, text=params[par].expr).grid(column=6, row=i)
        Label(paramdef, text=params[par].brute_step).grid(column=7, row=i)
        
        i=i+1

paramsbtn=Button(paramdef, text='Pick Params File', command=paramsbtnclick)
paramsbtn.grid(column=0,row=0)

def modelbtnclick():
    modelfilename=filedialog.askopenfilename(filetypes = (("TXT","*.txt"),("all files","*.*")))
    with open(modelfilename) as modelfile:
        modelreader = csv.reader(modelfile,delimiter='=')

        #Load rows into ODEtext dictionary, parsing the Snoopy output for dictionary indexes
        for row in modelreader:
            if row:
                index=row[0].replace("dc_","")
                index=index.replace("_/dt","")
                index=index.rstrip()
                ODEtext[index]=row[1]
        for eq in ODEtext:
            #cleanup text to match compartment dictionary
            for cindex in ODEtext:
                ODEtext[eq]=ODEtext[eq].replace("c_" + cindex + "_", "c['" + cindex + "']")
            #cleanup text to match parameters dictionary
            for pindex in params:
                ODEtext[eq]=ODEtext[eq].replace("p_" + pindex + "_", "p['" + pindex + "']")
            
    i=1
    global x0
    init = [0] * len(ODEtext)
    init[0] = 4500
    x0=tuple(init)
    print (x0)
    for k,v in ODEtext.items():
        check=IntVar()
        validboxes.append(Checkbutton(moddef,variable=check))
        validboxes[i-1].var=check
        validboxes[i-1].grid(column=1,row=i)
        Label(moddef, text='d' + k + 'dt').grid(column=0, row=i)
        Label(moddef, text=v,anchor='w',wraplength=720,width=120).grid(column=2, row=i)
        i+=1

modelbtn=Button(moddef, text='Pick Model File', command=modelbtnclick)
modelbtn.grid(column=0,row=0)

def f(y,t,ps):
    for par in ps:
        p[par]=ps[par].value
    i=0
    for k in ODEtext:
        c[k]=y[i]
        i+=1
    ODEs=[]
    for k,v in ODEtext.items():
        ODEs.append(eval(v))
    return ODEs

#solve the ODE model
def g(t, x0, ps):
    x=odeint(f,x0,t,args=(ps,))
    return x

#objective function (fitting function)
def residual (ps,ts,data):
    model=g(ts,x0,ps)
    #rewrite residuals without missing valid data
    i=len(validboxes)
    for item in reversed(validboxes):
        i=i-1
        if item.var.get() == 0:
            model=np.delete(model,i,axis=1)
    return(model-data).ravel()

#Validation Results Definition

def pickvalid():
    #getfile
    compartments=[]
    validfilename=filedialog.askopenfilename(filetypes = (("CSV","*.csv"),("all files","*.*")))
    with open(validfilename, newline='') as validfile:
        validreader = csv.reader(validfile)
        global t
        global expdata
        t=np.loadtxt(next(validreader),dtype='int16',delimiter=',')
        expdata=np.zeros(shape=(0,t.size))
        rowdata=np.zeros(shape=t.size)
        r=0
        for row in validreader:
            c=0
            for mass in row:
                if c==0:
                    compartments.append(mass)
                else:
                    rowdata[c-1]=float(mass)
                c=c+1
            expdata=np.append(expdata,[rowdata],axis=0)
            r=r+1

    #show valid results
    c=1
    Label(validdef, text='Compart', width=10, justify='left').grid(column=0,row=2)
    Label(validdef, text='Time', width=10, justify='left').grid(column=1,columnspan=t.size,row=1)
    for time in t:
        Label(validdef, text=time, width=10, justify='left').grid(column=c,row=2)
        c=c+1
    r=3
    for comp in compartments:
        Label(validdef, text=comp, width=10, justify='left').grid(column=0,row=r)
        r=r+1
    r=3
    for comp in expdata:
        c=1
        for mass in comp:
            Label(validdef, text=mass, width=10, justify='left').grid(column=c,row=r)
            c=c+1
        r=r+1

pickvalidbtn = Button(validdef, text="Pick Valid File", command=pickvalid)
pickvalidbtn.grid(column=0,row=0)


#Parameter Optimization
def savebtnclick():
    savefilename=filedialog.asksaveasfilename(filetypes = (("CSV","*.csv"),("all files","*.*")))
    with open(savefilename, 'w', newline='') as savefile:
        writer=csv.writer(savefile)
        writer.writerow(["Optimization Method",(minmethod)])
        writer.writerow(["Number of Evals",(result.nfev)])
        writer.writerow(["Chi-Square",(result.chisqr)])
        writer.writerow(["Reduced Chi-Square",(result.redchi)])
        writer.writerow(["Akaike Information Criterion",(result.aic)])
        writer.writerow(["Bayesian Information Criterion",(result.bic)])
        writer.writerow(["Name","Value","StdErr"])
        for param in result.params:
            writer.writerow([param,result.params[param].value,result.params[param].stderr])
    
def pobtn1click():
    # fit model and find predicted values
    global minmethod
    minmethod=methodcombo.get()
    data=np.transpose(expdata)
    global result
    result = minimize(residual, params, args=(t, data), method=minmethod)
    final = data + result.residual.reshape(data.shape)

    fig=plt.figure()
    fig.add_subplot(111)
    plt.plot(t,data,'o')
    plt.plot(t, final, '--', linewidth=2, c='blue')

    canvas = FigureCanvasTkAgg(fig,paramopt)
    canvas.show()
    canvas._tkcanvas.grid(column=0,row=1,columnspan=5,rowspan=25)
    
    # display fitted statistics
    Label(paramopt, text='Optimization Method').grid(column=0,row=26)
    Label(paramopt, text=minmethod).grid(column=1,row=26)
    Label(paramopt, text='Number of Evals').grid(column=0,row=27)
    Label(paramopt, text=result.nfev).grid(column=1,row=27)
    Label(paramopt, text='Chi-Square').grid(column=0,row=28)
    Label(paramopt, text=result.chisqr).grid(column=1,row=28)
    Label(paramopt, text='Reduced Chi-Square').grid(column=0,row=29)
    Label(paramopt, text=result.redchi).grid(column=1,row=29)
    Label(paramopt, text='Akaike Information Criterion').grid(column=0,row=30)
    Label(paramopt, text=result.aic).grid(column=1,row=30)
    Label(paramopt, text='Bayesian Information Criterion').grid(column=0,row=31)
    Label(paramopt, text=result.bic).grid(column=1,row=31)
    #report_fit(result)
    Label(paramopt, text='Name').grid(column=5, row=0)
    Label(paramopt, text='Value').grid(column=6, row=0)
    Label(paramopt, text='StdErr').grid(column=7, row=0)
    i=1
    for param in result.params:
        Label(paramopt, text=param).grid(column=5, row=i)
        Label(paramopt, text=result.params[param].value).grid(column=6, row=i)
        Label(paramopt, text=result.params[param].stderr).grid(column=7, row=i)
        i=i+1
    savebtn=Button(paramopt, text="Save Output", command=savebtnclick)
    savebtn.grid(column=2,row=0)
    
    
pobtn1 = Button(paramopt, text="Optimize", command=pobtn1click)
pobtn1.grid(column=1,row=0)

methodcombo= Combobox(paramopt)
methodcombo['values']=("leastsq","lbfgsb","cg","cobyla","tnc","slsqp","brute","differential_evolution","emcee")
methodcombo.current(0)
methodcombo.grid(column=0,row=0)

tab_control.pack(expand=1, fill='both')

window.mainloop()
