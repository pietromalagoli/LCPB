import mesa_web as mw
import pandas as pd 
import matplotlib.pyplot as plt

filename='StellarTracks/MESA-Web_M1_Z0001/profile1.data'
data=mw.read_profile(filename)

profile1_df=pd.DataFrame(data)
column_filter = ['mass','radius', 'initial_mass', 'initial_z', 'star_age', 'logRho','logT','Teff','energy','photosphere_L', 'photosphere_r', 'star_mass','h1','he3','he4']

# Create a new DataFrame with only the selected columns
filtered_profile1_df = profile1_df[column_filter].copy()

#normalize the radius
normalized_radius= filtered_profile1_df['radius'] / filtered_profile1_df['photosphere_r']
filtered_profile1_df['r_normalized'] = normalized_radius
#print (filtered_profile1_df)

#PLOT P1 LOGRHO vs RADIUS and MASS
#Plotting logRho against r_normlaized
plt.plot(filtered_profile1_df['r_normalized'], filtered_profile1_df['logRho'], label = 'logRho')
plt.xlabel('Normalized Radius')
plt.ylabel('Density (logRho)')
plt.title('(P1)Density Profile vsr_normalized')
plt.grid(True)
plt.legend()
plt.show()
#Plotting logRho against r_log
plt.plot(filtered_profile1_df['radius'], filtered_profile1_df['logRho'], label = 'logRho')
plt.xlabel('Radius (log scale)')
plt.ylabel('Density (logRho)')
plt.title('(P1)Density Profile vs r(log)')
plt.xscale('log')
plt.grid(True)
plt.legend()
plt.show()
# Plotting logRho against mass
plt.figure(figsize=(8, 6))
plt.plot(filtered_profile1_df['mass'], filtered_profile1_df['logRho'], label='logRho')
plt.xlabel('Mass')
plt.ylabel('logRho (Density)')
plt.title('(P1)Density Profile vs. Mass')
plt.grid(True)
plt.legend()
plt.show()

#PLOT P1 LOGT vs RADIUS and MASS
#logT vs r_normalized
plt.figure(figsize=(8,6))
plt.plot(filtered_profile1_df['r_normalized'], filtered_profile1_df['logT'], label='logT')
plt.xlabel('Normalized Radius')
plt.ylabel('Temperature (log)')
plt.title('(P1) Temperature Profile vs r_normalized')
plt.legend()
plt.grid(True)
plt.show()
# logT vs r_log
plt.figure(figsize=(8,6))
plt.plot(filtered_profile1_df['radius'], filtered_profile1_df['logT'], label='logT')
plt.xlabel('Radius (log scale)')
plt.ylabel('Temperature (log)')
plt.title('(P1) Temperature Profile vs r (log scale)')
plt.xscale('log')
plt.legend()
plt.grid(True)
plt.show()
#logT vs mass
plt.figure(figsize=(8,6))
plt.plot(filtered_profile1_df['mass'], filtered_profile1_df['logT'], label='logT')
plt.xlabel('MAss')
plt.ylabel('Temperature (log)')
plt.title('(P1) Temperature Profile vs Mass')
plt.legend()
plt.grid(True)
plt.show()