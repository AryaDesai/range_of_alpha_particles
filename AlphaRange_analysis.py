"""I (Arya Desai) wrote this code for the Alpha Range experiment in PHYS 403.This was developed throughout the course of the experiment as my desire
to use OriginPro was very low. I think my wrist actually hurts from all the clicking I had to do in OriginPro. I have tried to comment the code as much as possible
I have tried to make the code so that you can just enter the data from the MCA and it will output the plots and fits.
The code is pretty simple and modular so I hope it is understandable and if you have any questions, 
please feel free to reach out to me at aryad2@illinois.edu. I can also provide sample data if you want to test the code."""

#import statements
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from sklearn.metrics import r2_score
from scipy.integrate import quad

'''-------------------------FUNCTIONS-----------------------'''

#I wrote the first chunk of functions early on to convert the raw data we collect from the MCA to physically meaningful quantities we can use for fitting.

#energy is the energy of the alpha particle. Converted to MeV from centroid channel in MCA where the alpha particles are detected.
# The specific values of the slope(m) and intercept(o) in energy are determined by fitting a linear function to the data we collect from the MCA.
def energy(channel, m, o):
    e = (channel - o)/m
    return e

#distance is the distance from the source to the detector. Converted to cm from scale on back of apparatus
def distance(s):
    return (s-6.58)  

#Effective Distance is the range of the alpha particle at STP. Derived from P1V1 = P2V2
def X_eff(p,d,patm):
    return p*d/patm

#linear fit
def lin_fit(x,m,b):
    return m*x + b

#Define functions to calculate error. This is designed so I can just put in data 
# we collect from the sensor along with data created after passing raw data to the functions above and get the error values for all quantities.
def X_eff_error(P, D, patm, Delta_P, Delta_D):
    """
    Calculate the error in effective distance (X_eff).

    Parameters:
    P (float): Pressure measurement.
    D (float): Distance measurement.
    patm (float): Atmospheric pressure (constant).
    Delta_P (float): Error in pressure measurement.
    Delta_D (float): Error in distance measurement.

    Returns:
    float: Error in effective distance.
    """
    X_eff = P * D / patm
    Delta_X_eff = X_eff * np.sqrt((Delta_P / P) ** 2 + (Delta_D / D) ** 2)
    return Delta_X_eff
def energy_error(channel, Delta_channel, m):
    """
    Calculate the error in energy based on the error in channel reading.

    Parameters:
    channel (float): Channel measurement.
    Delta_channel (float): Error in channel measurement.
    m (float): Slope in the energy conversion equation.

    Returns:
    float: Error in energy.
    """
    Delta_E = Delta_channel / m
    return Delta_E

def net_rate_error(Counts, Integration_time, Delta_Counts, Delta_Integration_time):
    """
    Calculate the error in net rate based on the errors in counts and integration time.

    Parameters:
    Counts (float): The number of counts.
    Integration_time (float): Integration time in seconds.
    Delta_Counts (float): Error in counts.
    Delta_Integration_time (float): Error in integration time.

    Returns:
    float: Error in net rate.
    """
    Net_rate = Counts / Integration_time
    Delta_Net_rate = np.sqrt(Net_rate)/Integration_time
    return Delta_Net_rate

'''-------------------------RAW DATA-----------------------'''''

#This is where we would collect the data. Jessie would run the MCA software and I would note down the values it outputted here.
Channel = []
FWHM = [] #error in channel
Counts = []
Net_rate = [] #Counts per second
Pressure = [] #mm of Hg
Pressure_error = [] #error in Pressure 
d = [1] #scale in arbitrary units, signifies distance between source and detector
gas = ['example'] #gas used in chamber, just for my reference and book-keeping

#I just print this for a sanity check since the above columns rely on manual input of data and if they aren't the same length, then the code will crash.
# Although you could remove all of that above and just pass a csv of that structure to the df dataframe and it will work just fine.
print(len(Channel),len(FWHM),len(Counts),len(Net_rate),len(Pressure),len(Pressure_error),len(d))


'''-----------Use the functions defined above to convert raw data to physically meaningful quantities.--------------------'''

#I start by making a Pandas dataframe and store all the raw data from above in said dataframe. A dataframe is just a numy array which is a convenient 
# way to store data since you can perform array operations, which vectorizes the operations and makes them faster since there is no iteration and just 
# parallel computation. It also has a lot of built in functions which are useful for data analysis. And it can directly be outputted as a csv or even a
# latex table.

df = pd.DataFrame({'Channel': Channel, 'FWHM': FWHM, 'Counts': Counts, 'Pressure': Pressure,'Pressure_error': Pressure_error ,'Net rate': Net_rate, 'scale': d})
df['Energy(MeV)'] = energy(df['Channel'], 519.1, -24.05) #converts channel to energy of alpha particle in MeV. 
df['Distance(cm)'] = distance(df['scale']) #converts scale to distance in cm
df['Integration time(s)'] = df['Counts']/df['Net rate'] #Integration time in seconds, the data collection software is slow so the integration time is not constant and instead derived from the counts and net rate
df['effective distance'] = X_eff(df['Pressure'], df['Distance(cm)'], 763) #effective distance/range in cm
df['Distance error(cm)'] = df['effective distance']*0 + 0.13 #error in distance
df['Energy error(MeV)'] = energy_error(df['Channel'], df['FWHM'], 519.1) #error in energy
df['Net rate error'] = net_rate_error(df['Counts'], df['Integration time(s)'], np.sqrt(df['Counts']), df['Integration time(s)']*0 + 0.01) #error in net rate
df['X_eff error'] = X_eff_error(df['Pressure'], df['Distance(cm)'], 763, df['Pressure_error'], df['Distance error(cm)']) #error in effective distance

#output dataframe as csv. So, now you will have a csv with all these quantities and their errors. 
#everytime I add values to the raw data and rerun the code, the same csv is updated with the new values. This way, we don't have multiple files 
# and in case my code crashes or I forget to save, the data isn't lost.
df.to_csv('name_of_file_you_want).csv') 

'''------------------------PLOTTING AND FITTING-------------------'''

dff = df.loc[df['scale'] == 10.5] #filtering the dataframe to only include data for the same distance, since we can either vary pressure or distance and here we vary pressure.
df_filtered = dff.loc[df['Energy(MeV)'] > 1.5] #Since this is the linear fit, we filter out the low energy regime where the data isn't linear.

'''------------------------Linear Fit-------------------'''
#curve_fit performes a non-linear least squares fit to a function. It is called as curve_fit(function, xdata, ydata, p0 = initial_guess_for_parameters if you have any.).
# it returns two arrays, the first one is the optimized parameters(popt) and the second one is the covariance of the parameters(pcov).
# popt is calculated by minimizing the sum of the squared residuals of f(xdata, *popt) - ydata. where *popt is the unpacked tuple of parameters.
# pcov is the estimated covariance of popt. The diagonals provide the variance of the parameter estimate. 
# To compute one standard deviation errors on the parameters use perr = np.sqrt(np.diag(pcov)).
popt, pcov = curve_fit(lin_fit, df_filtered['effective distance'], df_filtered['Energy(MeV)']**2,) 

#Here I define y_fit as the fitted function with the optimized parameters. 
# I also calculate the R^2 value for the fit using the r2_score function from sklearn, another useful package for statistics.
y_fit = lin_fit(df_filtered['effective distance'], *popt)
r2 = r2_score(df_filtered['Energy(MeV)']**2, y_fit)

#Here I just plot the data and fit.

#plt.figure creates a figure object. figsize is the size of the figure in inches.A figure object is like a canvas on which you plot. 
# This is done to plot many things on the same figure, you could also use subplots but I prefer this method. 
plt.figure(figsize=(10,8))

#plt.scatter makes a scatter plot. The first argument is the x-axis data, the second argument is the y-axis data, 
# marker is the shape of the marker. Label is what will show up in the legend. There are many other arguments you can use.
plt.scatter(df_filtered['effective distance'],df_filtered['Energy(MeV)']**2,marker = 'o',color='blue',label='data')

# plt.errorbar plots errorbars. The first argument is the x-axis data, the second argument is the y-axis data, then you have to specify the x and y errors.
# fmt is the format of the errorbars, ecolor is the color of the errorbars, capsize is the size of the caps on the errorbars. 
plt.errorbar(df_filtered['effective distance'],df_filtered['Energy(MeV)']**2, xerr = df_filtered['X_eff error'], yerr = df_filtered['Energy error(MeV)'], fmt='none', ecolor='black', capsize=2)

# plt.plot plots a line. The first argument is the x-axis data, the second argument is the y-axis data, r- is the color and style of the line,
# the reason the label is written like this is because I want it to update based on what the data is
# So what it does is, %5.3f  takes the first 5 digits of the number and then rounds it to 3 decimal places. and the
# %(*popt, r2) unpacks the tuple of parameters and R^2 value. So, the label will be updated based on the values of the parameters and R^2 value.
# You need to make sure that the number of %f's is equal to the number of parameters you have in the label.
plt.plot(df_filtered['effective distance'], y_fit, 'r-', label='fit: m=%5.3f, b=%5.3f, R$^2$=%5.3f' % (*popt, r2))

# plt.xlabel and plt.ylabel are self explanatory. The r'' string is for collect latex formatting. 
# A neat thing about matplotlib is that it can use latex formatting for everything by using $your latex$.
plt.xlabel(r'X$_{eff}$ (cm)',fontsize = 18)
plt.ylabel('E$^2$ (MeV$^2$)',fontsize = 18)
plt.title('Energy$^2$ vs X$_{eff}$ for Ar',fontsize = 18)
plt.grid('True')
plt.legend()
plt.savefig('E vs Xeff day5(Ar)_fitted.png',dpi = 500, bbox_inches='tight')

#Here I just print the parameters and R^2 value for the fit for my reference. 
# I also calculate the x-intercept of the fit which is the range of the alpha particle at STP

print(f"R^2 value for the fit: {r2}")
print(f"x-intercept: {-popt[1]/popt[0]} cm")
print(r'The error in energy is on average', np.median(df_filtered['Energy error(MeV)']), 'MeV', 'and hence it is negligible')
print(r'The error in effective distance is on average', np.median(df_filtered['X_eff error']), 'cm', 'and hence it is negligible')

'''------------------NON LINEAR FIT-------------------'''

#Here I define the function I want to fit.
# This function models counts of the alpha particles as a function of pressure.
# I didn't write this above with the other functions since it is a bit more complicated and I wanted to have it where I use it so I wouldn't have to go up and down all the time.
def n(p, n0, p0, alpha):
    """
    Calculate the number of counts as a function of pressure. This is an erf function.

    Parameters: 
    p (tuple of floats): Pressure. 
    n0 (float): Number of counts at zero pressure.
    p0 (float): Pressure at which the number of counts is 0.
    alpha (float): Straggling parameter.
    """
    # integral here is the integrand term. I define it separately since it is complicated and its easier to read this way.
    # quad is a function from scipy.integrate which performs a quadrature integration.
    # I use a lambda function since it is more efficient and I don't have to define a function and then call it. 
    # So the line lambda x: blah, means the function takes x as an argument and returns blah.
    integral = lambda x: 1 / (alpha * np.sqrt(np.pi)) * np.exp(-((x - p0) / alpha)**2)

    # result is the integral of the integrand from 0 to p. p.apply applies the function to every element in the series p. qua
    # quad returns a tuple of the value of the integral and the error in the integral. It is defined as quad(function, lower limit, upper limit). 
    # I only want the value of the integral so I take the 0th element of the tuple.
    result = p.apply(lambda x: quad(integral, 0, x)[0])

    # return the number of counts as a function of pressure.
    return n0 * (1 - result)
# p_data and n_data are defined as the pressure and net rate data from the dataframe.
# I just made new variables since I didn't want to type df['Pressure'] and df['Net rate'] every time.
# but it is certainly a bit more inefficient since I am making new variables and using more memory. Shouldn't be a problem for small dataframes like this. 
p_data = df['Pressure']
n_data = df['Net rate']

# Initial guess for the parameters n0,p0, and alpha. 
# n0 is the number of counts at zero pressure, p0 is the pressure at which the number of counts is 0, and alpha is 0.0255*p0 from literature.
initial_guess = [25,800 ,15 ]

# Perform the curve fitting
params, _ = curve_fit(n, p_data, n_data, p0=initial_guess)

# Extract the parameters. x,y,z = p is called unpacking, where p is [_,_,_] array. It is a neat trick to assign multiple variables at once.
n0_opt, p0_opt, alpha_opt = params
r2_n = r2_score(n_data, n(p_data, *params))
# Print the parameters for sanity check
print(f"parameters: n0 = {n0_opt}, p0 = {p0_opt}, alpha = {alpha_opt}")

# Plot the data and the fit, similarly to the linear fit.
plt.figure(figsize=(10,8))
plt.scatter(df['Pressure'], df['Net rate'], color='purple', marker='o', label='data')
plt.errorbar(df['Pressure'], df['Net rate'], xerr=df['X_eff error'], yerr=df['Pressure_error'], fmt='none', ecolor='black')
plt.plot(df['Pressure'], n(df['Pressure'], *params), 'r-', label=r'fit: $n_0$=%5.3f, $p_0$=%5.3f, $\alpha$ =%5.3f, $r^2$=%5.3f'  % (*params,r2_n))
plt.xlabel('Pressure (mm of Hg)', fontsize=18)
plt.ylabel('Net rate', fontsize=18)
plt.title('Net rate vs Pressure for Ar', fontsize=18)
plt.grid(True)
plt.legend()
# plt.savefig saves the figure as a png, but you can use any format you want(like svg or jpeg) just write it in the filename, dpi is the dots per inch, bbox_inches='tight' makes sure the figure is saved with no extra whitespace.
plt.savefig('Net rate vs Pressure day5(Ar)_fitted.png', dpi=500, bbox_inches='tight')

#print various quantities for my reference.
print(r"$X_eff$ =",835.925*3.92/763,"cm")
print(r"X_eff error =",835.925*0.01/763,"cm")
print(r"Error in net rate =",np.mean(df['Net rate error']),"counts/s")
print(r'Here,$ n_0 $ is the maximum number of alpha particles that can be detected, $p_0$ is the pressure at which the net rate goes to 0 and $\alpha$ is the straggling parameter given by 0.015$p_0$, derived from literature')
