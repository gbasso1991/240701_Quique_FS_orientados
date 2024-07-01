#%% Analisis de los ciclos para calculo de tau  
import numpy as np
import matplotlib.pyplot as plt
import fnmatch
import os
import pandas as pd
import chardet 
import re
from scipy.interpolate import interp1d
#%% LECTOR RESULTADOS
def lector_resultados(path): 
    '''
    Para levantar archivos de resultados con columnas :
    Nombre_archivo	Time_m	Temperatura_(ºC)	Mr_(A/m)	Hc_(kA/m)	Campo_max_(A/m)	Mag_max_(A/m)	f0	mag0	dphi0	SAR_(W/g)	Tau_(s)	N	xi_M_0
    '''
    with open(path, 'rb') as f:
        codificacion = chardet.detect(f.read())['encoding']
        
    # Leer las primeras 6 líneas y crear un diccionario de meta
    meta = {}
    with open(path, 'r', encoding=codificacion) as f:
        for i in range(6):
            line = f.readline()
            if i == 0:
                match = re.search(r'Rango_Temperaturas_=_([-+]?\d+\.\d+)_([-+]?\d+\.\d+)', line)
                if match:
                    key = 'Rango_Temperaturas'
                    value = [float(match.group(1)), float(match.group(2))]
                    meta[key] = value
            else:
                match = re.search(r'(.+)_=_([-+]?\d+\.\d+)', line)
                if match:
                    key = match.group(1)[2:]
                    value = float(match.group(2))
                    meta[key] = value
                    
    # Leer los datos del archivo
    data = pd.read_table(path, header=14,
                         names=('name', 'Time_m', 'Temperatura',
                                'Remanencia', 'Coercitividad','Campo_max','Mag_max',
                                'frec_fund','mag_fund','dphi_fem',
                                'SAR','tau',
                                'N','xi_M_0'),
                         usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13),
                         decimal='.',
                         engine='python',
                         encoding=codificacion)
        
    files = pd.Series(data['name'][:]).to_numpy(dtype=str)
    time = pd.to_datetime(data['Time_m'][:],dayfirst=True)
    # delta_t = np.array([dt.total_seconds() for dt in (time-time[0])])
    temperatura = pd.Series(data['Temperatura'][:]).to_numpy(dtype=float)
    
    Mr = pd.Series(data['Remanencia'][:]).to_numpy(dtype=float)
    Hc = pd.Series(data['Coercitividad'][:]).to_numpy(dtype=float)
    campo_max = pd.Series(data['Campo_max'][:]).to_numpy(dtype=float)
    mag_max = pd.Series(data['Mag_max'][:]).to_numpy(dtype=float)
    
    xi_M_0=  pd.Series(data['xi_M_0'][:]).to_numpy(dtype=float)
     
    SAR = pd.Series(data['SAR'][:]).to_numpy(dtype=float)
    tau = pd.Series(data['tau'][:]).to_numpy(dtype=float)
   
    frecuencia_fund = pd.Series(data['frec_fund'][:]).to_numpy(dtype=float)
    dphi_fem = pd.Series(data['dphi_fem'][:]).to_numpy(dtype=float)
    magnitud_fund = pd.Series(data['mag_fund'][:]).to_numpy(dtype=float)
    
    N=pd.Series(data['N'][:]).to_numpy(dtype=int)
    return meta, files, time,temperatura,  Mr, Hc, campo_max, mag_max, xi_M_0, frecuencia_fund, magnitud_fund , dphi_fem, SAR, tau, N
#%% LECTOR CICLOS
def lector_ciclos(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()[:6]

    metadata = {'filename': os.path.split(filepath)[-1],
                "Concentracion_g/m^3": float(lines[0].strip().split('_=_')[1].split(' ')[0]),
            "C_Vs_to_Am_M": float(lines[1].strip().split('_=_')[1].split(' ')[0]),
            "pendiente_HvsI ": float(lines[2].strip().split('_=_')[1].split(' ')[0]),
            "ordenada_HvsI ": float(lines[3].strip().split('_=_')[1].split(' ')[0]),
            'frecuencia':float(lines[4].strip().split('_=_')[1].split(' ')[0])}
    
    data = pd.read_table(os.path.join(os.getcwd(),filepath),header=7,
                        names=('Tiempo_(s)','Campo_(kA/m)','Magnetizacion_(A/m)'),
                        usecols=(0,3,4),
                        decimal='.',engine='python',
                        dtype={'Tiempo_(s)':'float','Campo_(kA/m)':'float','Magnetizacion_(A/m)':'float'})  
    t= pd.Series(data['Tiempo_(s)']).to_numpy()
    H = pd.Series(data['Campo_(kA/m)']).to_numpy(dtype=float)*1000 #A/m
    M= pd.Series(data['Magnetizacion_(A/m)']).to_numpy(dtype=float)#A/m
    
    return t,H,M,metadata
#%% PROMEDIADOR
def promediador(directorio):
    for i,f in enumerate(os.listdir(directorio)):
        if i==0:
            t0,H0,M0,_=lector_ciclos(os.path.join(directorio,f))
        else:
            t_aux,H_aux,M_aux,_=lector_ciclos(os.path.join(directorio,f))
            t0+=t_aux
            H0+=H_aux
            M0+=M_aux
    t_prom=t0/(i+1)
    H_prom=H0/(i+1)
    M_prom=M0/(i+1)
    #print(directorio)
    print(f'''Promediados {i+1} archivos del directorio: 
    {directorio[-28:]}''')
    return t_prom,H_prom,M_prom
#%% TAU PROMEDIO
def Tau_promedio(filepath,recorto_extremos=20):
    '''Dado un path, toma archivo de ciclo M vs H
     Calcula Magnetizacion de Equilibrio, y Tau pesado con dM/dH
     '''
    t,H,M,meta=lector_ciclos(filepath)
     
    indx_max= np.nonzero(H==max(H))[0][0]
    t_mag = t[recorto_extremos:indx_max-recorto_extremos]
    H_mag = H[recorto_extremos:indx_max-recorto_extremos]
    M_mag = M[recorto_extremos:indx_max-recorto_extremos]

    H_demag = H[indx_max+recorto_extremos:-recorto_extremos] 
    # H_demag = np.concatenate((H_demag[:],H_mag[0:1]))

    M_demag = M[indx_max+recorto_extremos:-recorto_extremos]
    # M_demag = np.concatenate((M_demag[:],M_mag[0:1]))

    #INTERPOLACION de M 
    # Verificar que H_mag esté dentro del rango de H_demag
    #H_mag = H_mag[(H_mag >= min(H_demag)) & (H_mag <= max(H_demag))]

    # INTERPOLACION de M solo para los valores dentro del rango
    interpolador = interp1d(H_demag, M_demag,fill_value="extrapolate")
    M_demag_int = interpolador(H_mag)

    # interpolador=interp1d(H_demag, M_demag)
    # M_demag_int = interpolador(H_mag) 
    
    # Derivadas
    dMdH_mag = np.gradient(M_mag,H_mag)
    dMdH_demag_int = np.gradient(M_demag_int,H_mag)
    dHdt= np.gradient(H_mag,t_mag)

    Meq = (M_mag*dMdH_demag_int + M_demag_int*dMdH_mag)/(dMdH_mag+ dMdH_demag_int)
    dMeqdH = np.gradient(Meq,H_mag)

    Tau = (Meq - M_mag)/(dMdH_mag*dHdt )

    Tau_prom = np.sum(Tau*dMeqdH)/np.sum(dMdH_mag)
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    #%paso a kA/m y ns
    H_mag/=1e3
    H_demag/=1e3
    Tau *=1e9
    Tau_prom*=1e9
    print(meta['filename'])
    print(Tau_prom,'s')

    fig,(ax1,ax2) = plt.subplots(nrows=2,figsize=(7,6),constrained_layout=True)
    #ax1.plot(H,Tau,'-',label='U')
    ax1.plot(H_mag,Tau,'.-')
    ax1.grid()
    ax1.set_xlabel('H (kA/m)')
    ax1.set_ylabel(r'$\tau$ (s)')
    ax1.text(1/2,6/7,rf'<$\tau$> = {Tau_prom:.1f} ns',ha='center',va='center',
             bbox=dict(alpha=0.8),transform=ax1.transAxes,fontsize=11)

    ax1.grid()
    ax1.set_xlabel('H (A/m)')
    ax1.set_ylabel('$\\tau$ (ns)')
    ax1.set_title(r'$\tau$ vs H', loc='left')
    ax1.grid()

    ax2.plot(H_mag,Meq,'-',label='M$_{equilibrio}$')
    ax2.plot(H_mag,M_mag,label='Mag')
    ax2.plot(H_demag,M_demag,label='Demag')
    ax2.grid()
    ax2.legend()
    ax2.set_title('M vs H', loc='left')
    ax2.set_xlabel('H (kA/m)')
    ax2.set_ylabel('M (A/m)')

    axins = ax2.inset_axes([0.6, 0.12, 0.39, 0.4])
    axins.plot(H_mag,Meq,'.-')
    axins.plot(H_mag, M_mag,'.-')
    axins.plot(H_demag,M_demag,'.-')
    axins.set_xlim(-0.08*max(H_mag),0.08*max(H_mag)) 
    axins.set_ylim(-0.08*max(M_mag),0.08*max(M_mag))
    ax2.indicate_inset_zoom(axins, edgecolor="black")
    axins.grid()
    plt.suptitle(meta['filename'])

    return Meq , H_mag, max(H)/1000, Tau , Tau_prom , fig
#%% Axial
dir = os.path.join(os.getcwd(),'resultados_A','ciclos_promedio')
filepaths = [os.path.join(dir,f) for f in os.listdir(dir)]
filepaths.sort()

Mag_eqs = []
H_mags = []
Hmax_A=[]
taus_A=[]
tau_proms_A=[]
i=0
for f in filepaths:
    a,b,c,d,e,figura=Tau_promedio(f,recorto_extremos=50)
    Mag_eqs.append(a)
    H_mags.append(b)
    Hmax_A.append(c)
    taus_A.append(d)
    tau_proms_A.append(e)
    figura.savefig('tau_vs_H_FS_A_'+str(i)+'.png',dpi=300)
    i+=1

fig,(ax1,ax2) = plt.subplots(nrows=2,figsize=(8,7),constrained_layout=True,sharex=True)

for i,e in enumerate(taus_A):
    ax1.plot(H_mags[i],e,'.-',label=f'{Hmax_A[i]:.1f} kA/m',zorder=20-i)
ax1.grid()  

ax1.set_ylabel(r'$\tau$ (ns)')
ax1.set_title(r'$\tau$ vs H', loc='left')

ax1.legend(loc='center',ncol=5,bbox_to_anchor=(0, -0.85,1,1))

for i,e in enumerate(H_mags):
   
    ax2.plot(e,Mag_eqs[i],'.-',label='M$_{equilibrio}$'+str(i),zorder=20-i)
ax2.grid()
ax2.set_xticks(ticks=[-57.8,-53.3,-49.6,-45.9,-42.2,-38.5,-34.7,-31.0,-27.3,-23.6,-19.9,-16.2,-12.4,-8.7,-5.0,0,5.0,8.7,12.4,16.2,19.9,23.6,27.3,31.0,34.7,38.5,42.2,45.9,49.6,53.3,57.8],
                labels=['-57.8','-53.3','-49.6','-45.9','-42.2','-38.5','-34.7','-31.0','-27.3','-23.6','-19.9','-16.2','-12.4','-8.7','-5.0','0','5.0','8.7','12.4','16.2','19.9','23.6','27.3','31.0','34.7','38.5','42.2','45.9','49.6','53.3','57.8'],
                rotation=45,zorder=-4)

ax2.set_title('M$_{equilibrio}$ vs H', loc='left')
ax2.set_xlabel('H (kA/m)')
ax2.set_ylabel('M (A/m)')
plt.suptitle('FS - Ferrotec - Axial')
plt.savefig('tau_vs_H_FS_A_all.png',dpi=300)

fig,ax= plt.subplots(constrained_layout=True)
ax.plot(Hmax_A,tau_proms_A,'o-',label='Axial')
ax.set_title(r'$\tau$ vs H$_{max}$', loc='left')
ax.set_xlabel('H (kA/m)')
ax.set_ylabel(r'$\tau$ (ns)') 
ax.grid()
ax.set_xticks(ticks=[5.0,8.7,12.4,16.2,19.9,23.6,27.3,31.0,34.7,38.5,42.2,45.9,49.6,53.3,57.8],
                     labels=['5.0','8.7','12.4','16.2','19.9','23.6','27.3','31.0','34.7','38.5','42.2','45.9','49.6','53.3','57.8'],
               rotation=45)
plt.suptitle('FS - Ferrotec - Axial')
plt.savefig('tau_proms_vs_Hmax_A.png',dpi=300)
# Para guardarlos en un archivo ASCII con los datos en columnas, puedes hacer lo siguiente:
np.savetxt('tau_prom_vs_campo_max_A.txt', np.column_stack((Hmax_H,tau_proms_H)), 
           header='Hmax_(kA/m) | tau_prom_(ns) ' ,
           fmt=['%.2e','%.2e',], delimiter=' ', newline='\n')


#%% Perpendicular
dir = os.path.join(os.getcwd(),'resultados_P','ciclos_promedio')
filepaths = [os.path.join(dir,f) for f in os.listdir(dir)]
filepaths.sort()

Mag_eqs = []
H_mags = []
Hmax_P=[]
taus_P=[]
tau_proms_P=[]
i=0
for f in filepaths:
    a,b,c,d,e,figura=Tau_promedio(f,recorto_extremos=50)
    Mag_eqs.append(a)
    H_mags.append(b)
    Hmax_P.append(c)
    taus_P.append(d)
    tau_proms_P.append(e)
    figura.savefig('tau_vs_H_FS_P_'+str(i)+'.png',dpi=300)
    i+=1
    
#% Ciclo de Equilibrio 
fig,(ax1,ax2) = plt.subplots(nrows=2,figsize=(8,7),constrained_layout=True,sharex=True)
for i,e in enumerate(taus_P):
    ax1.plot(H_mags[i],e,'.-',label=f'{Hmax_P[i]:.1f} kA/m',zorder=20-i)
ax1.grid()
ax1.set_xlabel('H (kA/m)')
ax1.set_ylabel(r'$\tau$ (ns)')
ax1.set_title(r'$\tau$ vs H', loc='left')
ax1.legend(loc='center',ncol=5,bbox_to_anchor=(0, -0.85,1,1))

ax2.set_xticks(ticks=[-57.8,-53.3,-49.6,-45.9,-42.2,-38.5,-34.7,-31.0,-27.3,-23.6,-19.9,-16.2,-12.4,-8.7,-5.0,0,5.0,8.7,12.4,16.2,19.9,23.6,27.3,31.0,34.7,38.5,42.2,45.9,49.6,53.3,57.8],
                labels=['-57.8','-53.3','-49.6','-45.9','-42.2','-38.5','-34.7','-31.0','-27.3','-23.6','-19.9','-16.2','-12.4','-8.7','-5.0','0','5.0','8.7','12.4','16.2','19.9','23.6','27.3','31.0','34.7','38.5','42.2','45.9','49.6','53.3','57.8'],
                rotation=45,zorder=-4)

for i,e in enumerate(H_mags):
    ax2.plot(e,Mag_eqs[i],'.-',label='M$_{equilibrio}$'+str(i),zorder=20-i)
ax2.grid()
ax2.set_title('M$_{equilibrio}$ vs H', loc='left')
ax2.set_xlabel('H (kA/m)')
ax2.set_ylabel('M (A/m)')
plt.suptitle('FS - Ferrotec - Perpendicular')
plt.savefig('tau_vs_H_FS_P_all.png',dpi=300)

#Todos los tau
fig,ax= plt.subplots(constrained_layout=True)
ax.plot(Hmax_P,tau_proms_P,'o-')
ax.grid()
ax.set_xticks(ticks=[5.0,8.7,12.4,16.2,19.9,23.6,27.3,31.0,34.7,38.5,42.2,45.9,49.6,53.3,57.8],
                     labels=['5.0','8.7','12.4','16.2','19.9','23.6','27.3','31.0','34.7','38.5','42.2','45.9','49.6','53.3','57.8'],
               rotation=45)

ax.set_title(r'$\tau$ vs H$_{max}$', loc='left')
ax.set_xlabel('H (kA/m)')
ax.set_ylabel(r'$\tau$ (ns)') 
plt.suptitle('FS - Ferrotec - Perpendicular')
plt.savefig('tau_proms_vs_Hmax_P.png',dpi=300)

# Para guardarlos en un archivo ASCII con los datos en columnas, puedes hacer lo siguiente:
np.savetxt('tau_prom_vs_campo_max_P.txt', np.column_stack((Hmax_P,tau_proms_P)), 
           header='Hmax_(kA/m) | tau_prom_(ns) ' ,
           fmt=['%.2e','%.2e',], delimiter=' ', newline='\n')
#%% Random
dir = os.path.join(os.getcwd(),'resultados_R','ciclos_promedio')
filepaths = [os.path.join(dir,f) for f in os.listdir(dir)]
filepaths.sort()

Mag_eqs = []
H_mags = []
Hmax_R=[]
taus_R=[]
tau_proms_R=[]
i=0
for f in filepaths:
    a,b,c,d,e,figura=Tau_promedio(f,recorto_extremos=50)
    Mag_eqs.append(a)
    H_mags.append(b)
    Hmax_R.append(c)
    taus_R.append(d)
    tau_proms_R.append(e)
    figura.savefig('tau_vs_H_FS_R_'+str(i)+'.png',dpi=300)
    i+=1
    
#% Ciclo de Equilibrio 
fig,(ax1,ax2) = plt.subplots(nrows=2,figsize=(8,7),constrained_layout=True,sharex=True)
for i,e in enumerate(taus_R):
    ax1.plot(H_mags[i],e,'.-',label=f'{Hmax_R[i]:.1f} kA/m',zorder=20-i)
ax1.grid()
ax1.set_xlabel('H (kA/m)')
ax1.set_ylabel(r'$\tau$ (ns)')
ax1.set_title(r'$\tau$ vs H', loc='left')
ax1.legend(loc='center',ncol=5,bbox_to_anchor=(0, -0.85,1,1))

ax2.set_xticks(ticks=[-57.8,-53.3,-49.6,-45.9,-42.2,-38.5,-34.7,-31.0,-27.3,-23.6,-19.9,-16.2,-12.4,-8.7,-5.0,0,5.0,8.7,12.4,16.2,19.9,23.6,27.3,31.0,34.7,38.5,42.2,45.9,49.6,53.3,57.8],
                labels=['-57.8','-53.3','-49.6','-45.9','-42.2','-38.5','-34.7','-31.0','-27.3','-23.6','-19.9','-16.2','-12.4','-8.7','-5.0','0','5.0','8.7','12.4','16.2','19.9','23.6','27.3','31.0','34.7','38.5','42.2','45.9','49.6','53.3','57.8'],
                rotation=45,zorder=-4)

for i,e in enumerate(H_mags):
    ax2.plot(e,Mag_eqs[i],'.-',label='M$_{equilibrio}$'+str(i),zorder=20-i)
ax2.grid()
ax2.set_title('M$_{equilibrio}$ vs H', loc='left')
ax2.set_xlabel('H (kA/m)')
ax2.set_ylabel('M (A/m)')
plt.suptitle('FS - Ferrotec - Perpendicular')

plt.savefig('tau_vs_H_FS_P_all.png',dpi=300)
#Todos los tau
fig,ax= plt.subplots(constrained_layout=True)
ax.plot(Hmax_R,tau_proms_R,'o-')
ax.grid()
ax.set_xticks(ticks=[5.0,8.7,12.4,16.2,19.9,23.6,27.3,31.0,34.7,38.5,42.2,45.9,49.6,53.3,57.8],
                     labels=['5.0','8.7','12.4','16.2','19.9','23.6','27.3','31.0','34.7','38.5','42.2','45.9','49.6','53.3','57.8'],
               rotation=45)

ax.set_title(r'$\tau$ vs H$_{max}$', loc='left')
ax.set_xlabel('H (kA/m)')
ax.set_ylabel(r'$\tau$ (ns)') 
plt.suptitle('FS - Ferrotec - Random')
plt.savefig('tau_proms_vs_Hmax_R.png',dpi=300)

# Para guardarlos en un archivo ASCII con los datos en columnas, puedes hacer lo siguiente:
np.savetxt('tau_prom_vs_campo_max_R.txt', np.column_stack((Hmax_R,tau_proms_R)), 
           header='Hmax_(kA/m) | tau_prom_(ns) ' ,
           fmt=['%.2e','%.2e',], delimiter=' ', newline='\n')

# %% COMPARATIVA
fig,ax= plt.subplots(constrained_layout=True)
ax.plot(Hmax_P,tau_proms_P,'o-',label='Perpendicular')
ax.plot(Hmax_A,tau_proms_A,'s-',label='Axial')
ax.plot(Hmax_R,tau_proms_R,'v-',label='Random')

ax.legend()
ax.set_title(r'$\tau$ vs H$_{max}$', loc='center',fontsize=14)
ax.set_xlabel('H (kA/m)')
ax.set_ylabel(r'$\tau$ (ns)') 
ax.grid()
ax.set_xticks(ticks=[0,5.0,8.7,12.4,16.2,19.9,23.6,27.3,31.0,34.7,38.5,42.2,45.9,49.6,53.3,57.8],
                     labels=['0','5.0','8.7','12.4','16.2','19.9','23.6','27.3','31.0','34.7','38.5','42.2','45.9','49.6','53.3','57.8'],
               rotation=45)

plt.savefig('tau_vs_Hmax_comparativa_PAR.png',dpi=300)