import numpy as np
from new_dmp import DiscreteDMP
from lwr import LWR

def perform_new_dmp(given_traj, initial=None, end=None, duration=1.0, dt=0.001, use_improved=True):
    traj_1d = np.reshape(given_traj, np.size(given_traj))
    list_traj = traj_1d.tolist()
    if not initial:
        initial = list_traj[0]
    if not end:
        end = list_traj[-1]
        
    while (hasattr(initial, "__getitem__")):
        initial = initial[0]
    while (hasattr(end, "__getitem__")):
        end = end[0]
        
    traj_freq = int(duration / dt)
    
    traj = DiscreteDMP.compute_derivatives(list_traj, traj_freq)
    
    traj = list(traj)
    
    dmp = DiscreteDMP(improved_version=use_improved, reg_model=LWR(activation=0.1, exponentially_spaced=True, n_rfs=20))
    dmp.learn_batch(traj, traj_freq)
    
    dmp_adapt = DiscreteDMP(improved_version=use_improved, reg_model=dmp.lwr_model) #copy.deepcopy(dmp.lwr_model))
    dmp_adapt._is_learned = True  
    
    dmp.delta_t = dt
    dmp.setup(traj_1d[0], traj_1d[-1], duration)
    
    dmp_adapt.delta_t = dt  
    dmp_adapt.setup(initial, end, duration)
    
    traj_reproduced = []
    traj_adapted = []
    
    for _ in range(int(dmp.tau / dmp.delta_t)):
      dmp.run_step()
      dmp_adapt.run_step()
      traj_reproduced.append(dmp.x)
      traj_adapted.append(dmp_adapt.x)
      
    traj_reproduced = np.array(traj_reproduced).reshape(np.size(given_traj))
      
    print(np.size(traj_adapted))  
    traj_adapted = np.array(traj_adapted).reshape(np.size(given_traj))
    print(np.size(traj_adapted))  
    
    return [traj_adapted, traj_reproduced]
    
    
def perform_new_dmp_adapted(given_traj, initial=None, end=None, duration=1.0, dt=None, use_improved=True, k=None, d=None):
    traj_1d = np.reshape(given_traj, np.size(given_traj))
    list_traj = traj_1d.tolist()
    if not initial:
        initial = list_traj[0]
    if not end:
        end = list_traj[-1]
        
    if (isinstance(initial, (list, np.ndarray))):
        #print(initial)
        initial = initial[0]
    if (isinstance(end, (list, np.ndarray))):
        end = end[0]
        
    traj_freq = int(np.size(given_traj))
    
    if not dt:
        dt = duration / traj_freq
    
    traj = DiscreteDMP.compute_derivatives(list_traj, traj_freq)
    
    traj = list(traj)
    
    dmp = DiscreteDMP(improved_version=use_improved, reg_model=LWR(activation=0.1, exponentially_spaced=True, n_rfs=20), K=k, D=d)
    dmp.learn_batch(traj, traj_freq)
    
    dmp_adapt = DiscreteDMP(improved_version=use_improved, reg_model=dmp.lwr_model, K=k, D=d) #copy.deepcopy(dmp.lwr_model))
    dmp_adapt._is_learned = True  
    
    dmp.delta_t = dt
    dmp.setup(traj_1d[0], traj_1d[-1], duration)
    
    dmp_adapt.delta_t = dt  
    dmp_adapt.setup(initial, end, duration)
    
    traj_reproduced = []
    traj_adapted = []
    
    for _ in range(traj_freq):#(int(dmp.tau / dmp.delta_t)):
      dmp.run_step()
      dmp_adapt.run_step()
      traj_reproduced.append(dmp.x)
      traj_adapted.append(dmp_adapt.x)
      
      
    #print(np.size(traj_adapted))  
    traj_adapted = np.array(traj_adapted).reshape(np.size(given_traj))
    #print(np.size(traj_adapted))  
    
    #print(traj_adapted)
    
    return traj_adapted

def perform_dmp_general(given_traj, constraints, indeces, duration=1.0, dt=None, use_improved=True, k=None, d=None):
    adapted_traj = np.zeros(np.shape(given_traj))
    n_pts, n_dims = np.shape(given_traj)
    
    for d in range(n_dims):
        traj_1d = np.reshape(given_traj[:, d], (n_pts,))
        #list_traj = traj_1d.tolist()
        for ind in range(len(indeces)):
            if (indeces[ind] == 0):
                initp = constraints[ind][d]
                #print('Initial constraint: %f' % (initp))
            elif (indeces[ind] == n_pts - 1) or (indeces[ind] == - 1):
                endp = constraints[ind][d]
                #print('End constraint: %f' % (endp))
            else:
                print('WARNING: This implementation of DMP cannot via-point deform! Constraint not included.')
        traj_adapted = perform_new_dmp_adapted(traj_1d, initp, endp)
        #  
        #if (isinstance(initial, (list, np.ndarray))):
        #    print(initial)
        #    initial = initial[0]
        #if (isinstance(end, (list, np.ndarray))):
        #    end = end[0]
        #    
        #traj_freq = n_pts
        #
        #if not dt:
        #    dt = duration / traj_freq
        #
        #traj = DiscreteDMP.compute_derivatives(list_traj, traj_freq)
        #
        #traj = list(traj)
        #
        #dmp = DiscreteDMP(improved_version=use_improved, reg_model=LWR(activation=0.1, exponentially_spaced=True, n_rfs=20), K=70, D=d)
        #dmp.learn_batch(traj, traj_freq)
        #
        #dmp_adapt = DiscreteDMP(improved_version=use_improved, reg_model=dmp.lwr_model, K=k, D=d) #copy.deepcopy(dmp.lwr_model))
        #dmp_adapt._is_learned = True  
        #
        #dmp.delta_t = dt
        #dmp.setup(traj_1d[0], traj_1d[-1], duration)
        #
        #dmp_adapt.delta_t = dt  
        #dmp_adapt.setup(initial, end, duration)
        #
        #traj_reproduced = []
        #traj_adapted = []
        #
        #for _ in range(int(dmp.tau / dmp.delta_t)):
        #    dmp.run_step()
        #    dmp_adapt.run_step()
        #    traj_reproduced.append(dmp.x)
        #    traj_adapted.append(dmp_adapt.x)
        #
        traj_adapted = np.array(traj_adapted).reshape(np.shape(given_traj[:, d]))
        adapted_traj[:, d] = traj_adapted
        
    return adapted_traj
    
#def perform_dmp_improved(traj, initial=[], end=[], ay=None, by=None):
#  #set up endpoints if none specified
#  if not initial:
#    initial = traj[0]
#  if not end:
#    end = traj[max(np.shape(traj)) - 1]
#  #transpose if necessary
#  if len(np.shape(traj)) > 1:
#    if np.shape(traj)[0] > np.shape(traj)[1]:
#      traj = np.transpose(traj)
#  ntraj = np.reshape(traj, (1, max(np.shape(traj))))
#  ## DMP ##
#  dmp_traj = perform_dmp(ntraj, [initial, end], alpha_y=ay, beta_y=by)
#  dmp_traj = np.reshape(dmp_traj, np.size(traj))
#  return dmp_traj