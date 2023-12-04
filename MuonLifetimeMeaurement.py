import numpy as np
import matplotlib.pyplot as plt
import ROOT
from iminuit import Minuit

def generate_time(n1,tau):
    ran = ROOT.TRandom3(0)
    root_time_measurements = []
    for _ in range(n1):
        r1 = ran.Uniform(0.0, 1.0)
        r = -np.log(r1)/(1/tau)
        root_time_measurements.append(r)
    return root_time_measurements


def generate_time_numpy(n1,tau):
    numpy_time_measurements = np.random.exponential(scale=tau, size=n1)
    return numpy_time_measurements


def fill_tree_with_time(outfile, time_measurements):
    tree = ROOT.TTree("time_measurements", "Time Measurement Data")
    x = np.zeros(1, dtype=float)
    tree.Branch("x", x, "x/D")

    for r in time_measurements:
        x[0] = r
        tree.Fill()

    tree.Write()

    
def fit_time_measurements(outfile):
    tree = outfile.Get("time_measurements")

    def fcn(tau):
        xVec = [entry.x for entry in tree]
        lnL = 0.0
        for x in xVec:
            pdf = (np.exp(-x/tau))/tau
            if pdf > 0.0:
                lnL += np.log(pdf)
            else:
                print("pdf is negative!!!")
        f = -2.0 * lnL
        return f
    minuit = Minuit(fcn, tau=2)
    minuit.limits["tau"] = (2, 10)
    minuit.migrad()
    minuit.hesse()
    tau_expected = minuit.values["tau"]
    error_tau_expected = minuit.errors["tau"]
    print("Expected tau:", tau_expected, "+/-", error_tau_expected)
    return tau_expected

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
    plt.show()
    plt.close('all')


def plot_tau_expected(tau_expected_samples):
    plt.figure(figsize=(12, 6))
    plt.hist(tau_expected_samples, bins=50, density=True, alpha=0.7, label="tau expected")
    plt.xlabel("tau expected")
    plt.ylabel(" ")
    plt.title("Histogram of expected tau in many experiments")
    plt.legend()
    plt.show()

def check_error(tau_expected_samples,n1,n2,tau):
    #sample_mean_std1 = np.std(samples_mean)
    sum_squared_diff = np.sum((tau_expected_samples - np.mean(tau_expected_samples))**2)
    tau_expected_std = np.sqrt(sum_squared_diff/(n2-1))
    time_measurements_std = tau/(np.sqrt(n1))
    print(f"Error on the tau expected: {tau_expected_std:.4f}")
    print(f"Expected std of time measurements: {time_measurements_std:.4f}")
    print(f"Ratio: {tau_expected_std / time_measurements_std:.4f}")

###Main Code###
outfile = ROOT.TFile("DecayTimeMeasurements.root", "recreate")
n1 = 1000 #number of time measurements in a tauticular experiment
n2 = 10000 #total number of experiments
tau = 5 #mean muon lifetime

time_measurements = generate_time(n1,tau) #transforming U(0,1) to exponential random number t
#time_measurements = generate_time_numpy(n1,tau) #generating exponential random number t(0,infinity) using numpy
fill_tree_with_time(outfile, time_measurements) #filling tree with the time measuremnets

tau_expected = fit_time_measurements(outfile) #Unbinned NLL fit to the generated time_measurements data using Minuit

plot_time_measurement_fit(outfile, tau_expected) # plotting the fit along with data

tau_expected_samples = []
print("Fitting the samples")
for _ in range(n2):
    time_measurements_sample = np.array(generate_time(n1,tau))
    #time_measurements_samples = np.array(generate_time_numpy(n1,tau))# to use numpy
    outfileTemp = ROOT.TFile("DecayTimeMeasurementsSample.root", "recreate")
    fill_tree_with_time(outfileTemp, time_measurements_sample)
    tau_expected_sample  = fit_time_measurements(outfileTemp)
    tau_expected_samples.append(tau_expected_sample)
    
plot_tau_expected(tau_expected_samples)
check_error(tau_expected_samples, n1, n2, tau)
