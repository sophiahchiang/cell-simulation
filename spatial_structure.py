from functions_spatial import *

parser = argparse.ArgumentParser()
parser.add_argument("--net_size",type=int, required=True)
parser.add_argument("--iter",type=float, required=True)
parser.add_argument("--pyr_idx", type=int, required=True)
parser.add_argument("--n_reps", type=int, required=True)
#parser.add_argument("--fit_diff",'--list',nargs='+',required=True)
parser.add_argument("--mut_prob", '--file', type=str, nargs='+', action='append', required=True)

args = parser.parse_args()

N = args.net_size
t = int(args.iter)
replicates = args.n_reps

# Create grid
arr_zeros = np.zeros([N,N])
# Initialize grid
arr_0 = fill_circle(arr_zeros, 0.1)
# Initialize boundary array
arr = np.copy(arr_0)
bounds = init_boundary_arr(arr)

#fne = '/Users/sophiachiang/Documents/Documents/Murugan Lab/fitness_fitted.dat'
#pyr_level_idx = args.pyr_idx
#growth_rate_vec, f_318_WT, f_611_mut, f_611_WT = load_growth_rates(fne, pyr_level_idx)

# Set probability to mutate: p_mu = [mutator, non-mutator] in [0,1)
# mutation rate of the mutator is 4 orders of magnitude larger than for the non-mutator
# non-mutator almost never mutates


#redo graphs so that 0 is on smallest end, 0 first
fitness_vals = [[0, -0.001, -0.01, -0.1],
                [0, 0.001, 0.01, 0.1]] 

fig,axs = plt.subplots(1,4,figsize=(16,6))

print("Number of different mutation probability: ", len(args.mut_prob))
final_ratios = []
final_counts = []
#for j in range(len(fit_diff)):  
for index in range(len(args.mut_prob)):
        p_mu = args.mut_prob[index]
        #inter_ratios = []
        for i_df1 in range(len(fitness_vals[0])):
            ratios = []
            counts = []
            df1 = fitness_vals[0][i_df1]
            df2 = fitness_vals[1][i_df1]
            print("Our fitness differences are: ",df1,df2)
            NM, M = fitness_diff(df1, df2)
            growth_rate_vec = [0.5,NM,0,M,1]
            for i in range(replicates):
                print("Number of replicate: ", i)
                arr_zeros = np.zeros([N,N])
                arr_0 = fill_circle(arr_zeros, 0.1)
                arr = np.copy(arr_0)
                bounds = init_boundary_arr(arr)
                for tt in range(t):
                    rand_cell = random_b_cell(bounds)
                    x, y = rand_cell[0], rand_cell[1]
                    if flipg(x, y, arr, growth_rate_vec):
                        #fills all adjacent empty cells with grid[x,y]
                        children = divide(x, y, arr, bounds)
                        for c in range(len(children)):
                            if flipm(children[c][0], children[c][1], arr, p_mu):
                               arr = mutate(children[c][0], children[c][1], arr)
                    
                init_count, init_bounds, init_ratio = edge_ratio(init_boundary_arr(arr_0),arr_0)
                final_count, final_bounds, final_ratio = edge_ratio(bounds, arr)
                
                ratios.append(final_ratio)
                counts.append(final_count)
                    
            im = axs[i_df1].boxplot(ratios) 
            axs[i_df1].set_title('$\mu$: %f $\Delta F_1$: %s $\Delta F_2$: %s' % (float(p_mu[0]), "{:.3f}".format(df1),"{:.3f}".format(df2)), fontsize = 12)
            #axs[i_df1].set_ylim([0,max(ratios)+0.1*max(ratios)])
        fig.tight_layout()
        plt.savefig('histograms_%f.pdf' % (float(p_mu[0])))


