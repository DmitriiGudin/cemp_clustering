from __future__ import division
import numpy as np
import scipy.stats as ss
import itertools


filename = 'HDBSCAN_test.csv'

values = [0.5, 0.33, 0.25]
abundance_text = ['[Fe/H]', '[C/Fe]c', '[Sr/Fe]', '[Ba/Fe]']


def get_column(N, Type, filename):
    return np.genfromtxt(filename, delimiter=',', dtype=Type, skip_header=1, usecols=N, comments=None)


def get_cumul_binom_above(n, N, p): # n successes out of N trials with success probability of p
    if np.isnan(float(p)) or p<0 or p>1 or N<0 or n<0 or (not float(n).is_integer()) or (not float(N).is_integer()):
        return np.nan
    elif n>N:
        return 1
    else:
        hh = ss.binom(N, p)
        cumul_p = 0
        for k in range (n, N+1):
            cumul_p += hh.pmf(k)
        return cumul_p


def get_binom(n, N, p):
    if np.isnan(float(p)) or p<0 or p>1 or N<0 or n<0 or (not float(n).is_integer()) or (not float(N).is_integer()):
        return np.nan
    elif n>N:
        return 0
    else:
        hh = ss.binom(N,p)
        return hh.pmf(n)


def get_combined_cumul(Ns, N, values=values):
    total_p = 0
    Ns, values = sorted(Ns), sorted(values)
    for ns in sorted(list(set(list(itertools.product(range(N+1),repeat=len(Ns)))))):
        ns = list(ns)
        if ns == sorted(ns):
            inequalities_True = True
            for n1, n2 in zip (ns, Ns):
                if n1 < n2:
                    inequalities_True = False
            if inequalities_True:
                prod = get_binom(ns[0],N,values[0])
                if len(ns)>1:
                    for i in range(len(ns[1:])):
                        if np.isnan(get_binom(ns[i+1]-ns[i],N-ns[i], (values[i+1]-values[i])/(1-values[i]))):
                            prod *= 1
                        else:
                            prod *= get_binom(ns[i+1]-ns[i],N-ns[i], (values[i+1]-values[i])/(1-values[i]))
                total_p += prod
    return total_p
                

def get_tot_prob(p_array):
    return np.prod(max([p for p in p_array])**len(p_array))


def get_prob(prob, prec='.1f'):
    return format(prob*100, prec)


if __name__ == '__main__':
    N_FeH = get_column(1, int, filename)[0]
    N_CFec = get_column(2, int, filename)[0]
    N_SrFe = get_column(3, int, filename)[0]
    N_BaFe = get_column(4, int, filename)[0]

    FeH = get_column(1, int, filename)[1:]
    CFec = get_column(2, int, filename)[1:]
    SrFe = get_column(3, int, filename)[1:]
    BaFe = get_column(4, int, filename)[1:]

    prob_array = [[] for a in range(len(abundance_text))]

    for i, (n, a, a_t) in enumerate (zip([N_FeH, N_CFec, N_SrFe, N_BaFe], [FeH, CFec, SrFe, BaFe], abundance_text)):
        print a_t + ': ' 
        print str(n) + ' measurements.'
        for j, v in enumerate(values):
            print 'Above ' + str(v) + ': ' + str(a[j]) + ' for the cumulative binomial (IEAD) probability of ' + get_prob(get_cumul_binom_above(a[j], n, v)) + '%.'
            prob_array[i].append(get_cumul_binom_above(a[j], n, v))
        print ""
    print "----"
    print ""
    print "Total cumulative binomial (FEAD) probability: "
    for v, arr in zip(values, np.transpose(np.array(prob_array))):
        print "Below " + str(v) + ": " + get_prob(get_combined_cumul([1,2,3,4], 4, arr), '.10f') + "%"
    
    print ""
    print "----"
    print ""
    print "Combined binomial (GEAD) probabilities: "
    combined_probs = []
    for n, a, a_t in zip([N_FeH, N_CFec, N_SrFe, N_BaFe], [FeH, CFec, SrFe, BaFe], abundance_text):
        print a_t + ': ' + get_prob(get_combined_cumul(a, n, values)) + "%"
        combined_probs.append(get_combined_cumul(a, n, values))
    
    print ""
    print "----"
    print ""
    print "Ultimate probability: " + get_prob(get_combined_cumul([1,2,3,4],4,combined_probs), '.15f') + "%"

    prob_array = [[] for a in range(2)]
    for i, (n, a) in enumerate (zip([N_FeH, N_CFec], [FeH, CFec])):
        for j, v in enumerate(values):
            prob_array[i].append(get_cumul_binom_above(a[j], n, v))
    print ""
    #print "----"
    #print ""
    #print "Environmental ([Fe/H], [C/Fe]c) FEAD probabilities: "
    #for v, arr in zip(values, np.transpose(np.array(prob_array))):
    #    print "Below " + str(v) + ": " + get_prob(get_combined_cumul([1,2], 2, arr)) + "%"
    #print "Ultimate (OEAD) probability: " + get_prob(get_combined_cumul([1,2],2,combined_probs[:2]), '.10f') + "%"
    #
    #prob_array = [[] for a in range(3)]
    #for i, (n, a) in enumerate (zip([N_SrFe, N_BaFe], [SrFe, BaFe])):
    #    for j, v in enumerate(values):
    #        prob_array[i].append(get_cumul_binom_above(a[j], n, v))
    #print ""
    #print "----"
    #print ""
    #print "Progenitor ([Sr/Fe], [Ba/Fe]) FEAD probabilities: "
    #for v, arr in zip(values, np.transpose(np.array(prob_array))):
    #    print "Below " + str(v) + ": " + get_prob(get_combined_cumul([1,2], 2, arr)) + "%"
    #print "Ultimate (OEAD) probability: " + get_prob(get_combined_cumul([1,2],2,combined_probs[2:]), '.10f') + "%"
    
