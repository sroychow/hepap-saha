import numpy as np
import matplotlib.pyplot as plt
import ROOT
from iminuit import Minuit


# Part1: Create exponential random numbers for a given value of tau and plot it ;
# Exponential pdf = 1/tau*e^(-t/tau)

##transformi U(0,1) to exponential random number t
def generate_time(n1,tau):
    root_time_measurements = []
    #TODO:generate exponential random numbers->store in root_time_measurements
    #hints:You can use TRandom3() class from Root
    return root_time_measurements

##generating exponential random number t using numpy
def generate_time_numpy(n1,tau):
    #TODO:generate random number using numpy->store in numpy_time_measurements
    return numpy_time_measurements

##filling tree with the time measuremnets
def fill_tree_with_time(outfile, time_measurements):
    tree = ROOT.TTree("time_measurements", "Time Measurement Data")
    #TODO: Fill the tree branch with time measurements from outfile
    tree.Write()

##plot the time measurements
def plot_time_measurements(outfile):
    tree = outfile.Get("time_measurements")
    x_data = np.array([entry.x for entry in tree])
    plt.figure(figsize=(8, 6))
    plt.hist(x_data, bins=50, density=True, alpha=0.7, label="Generated Time Measurement Data")
    plt.xlabel("time")
    plt.ylabel("")
    plt.legend()
    plt.title("Generated Exponential Distribution")
    plt.savefig("TimeMeasurements.png")
    plt.show()
    plt.close('all')

outfile = ROOT.TFile("DecayTimeMeasurements.root", "recreate")
n1 = 1000 #number of time measurements in a particular experiment
tau = 2 #mean muon lifetime
time_measurements = generate_time(n1,tau) 
#time_measurements = generate_time_numpy(n1,tau) #generating exponential random number t using numpy
fill_tree_with_time(outfile, time_measurements) 
plot_time_measurements(outfile)

'''
# Part2: Fit exponential random numbers and plot it ;

##Unbinned NLL fit to the generated time_measurements data using Minuit
def fit_time_measurements(outfile):
    tree = outfile.Get("time_measurements")

    def fcn(tau):
        time = [entry.x for entry in tree]
        lnL = 0.0 
        for x in time:                                                                 
          #TODO: construct the log likelihood
        return -2.0*lnL
    minuit = Minuit(fcn, tau=2)
    minuit.limits["tau"] = (2, 10)
    minuit.migrad()
    minuit.hesse()
    tau_expected = minuit.values["tau"]
    error_tau_expected = minuit.errors["tau"]
    print("Expected tau:", tau_expected, "+/-", error_tau_expected)
    return tau_expected

##Plot the fit and the data
def plot_time_measurement_fit(outfile, tau_expected):
    tree = outfile.Get("time_measurements")
    x_data = np.array([entry.x for entry in tree])
    x_fit = np.linspace(0, max(x_data), 1000)
    y_fit = np.exp(-x_fit/tau_expected)/tau_expected
    plt.figure(figsize=(8, 6))
    plt.hist(x_data, bins=50, density=True, alpha=0.7, label="Generated Time Measurement Data")
    plt.plot(x_fit, y_fit, label="Fitted Exponential Distribution", color="red", linewidth=2)
    plt.xlabel("time")
    plt.ylabel("")
    plt.legend()
    plt.text(
        0.6,
        0.8,
        f'Fitted tau: {tau_expected:.4f}',
        transform=plt.gca().transAxes,
        bbox=dict(facecolor='white', alpha=0.8),
    )
    plt.title("Fitted Exponential Distribution")
    plt.savefig("TimeMeasurementsFit.png")
    plt.show()
    plt.close('all')
    
## Fit distribution and then plot the fit along with data
tau_expected = fit_time_measurements(outfile)
plot_time_measurement_fit(outfile, tau_expected)

# Part 3: Do n2 such experiments and plot the fitted taus ; Please have a look at the mean of the gaussian
# The Central Limit Theorem:
# For samples of size  30 or more, the sample mean is approximately normally distributed, with mean mu  and standard deviation = tau/sqrt(n); where tau = std of the sample, n  is the sample size. 

n2=1000 #total number of experiments
tau_expected_samples = []
print("Fitting the samples")
#TODO: Do n2 such experiments: for each experiment fit the n1 random number's  distribution->calculate the expected value of tau->store the expected values in tau_expected_sample

def plot_tau_expected(tau_expected_samples):
    plt.figure(figsize=(12, 6))
    plt.hist(tau_expected_samples, bins=25, density=True, alpha=0.7, label="tau expected")
    plt.xlabel("tau expected")
    plt.ylabel(" ")
    plt.title("Histogram of expected tau in many experiments")
    plt.legend()
    plt.savefig("ExpectedTauDist.png")
    plt.show()
    
plot_tau_expected(tau_expected_samples)

# Part 4: From the distribution of expected tau, calculate the std and match with the expected (sigma/sqrt(n)) one from CLT

def check_error(tau_expected_samples,n1,n2,tau):
    #TODO:Find the Error on tau expected from tau_expected_samples -> tau_expected_std
    time_measurements_std = tau/(np.sqrt(n1))
    print(f"Error on the tau expected: {tau_expected_std:.4f}")
    print(f"Expected std : {time_measurements_std:.4f}")
    print(f"Ratio: {tau_expected_std / time_measurements_std:.4f}")
check_error(tau_expected_samples, n1, n2, tau)

'''
