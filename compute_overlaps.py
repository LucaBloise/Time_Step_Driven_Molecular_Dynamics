import math
import sys

def compute():
    filename = r'outputs/scanningRate/scan_N100_k1000_seed825920765772900_20260517_101624.txt'
    
    L = 80.0
    r0 = 1.0
    r = 1.0
    
    max_pp_overlap = 0.0
    max_pp_time = 0.0
    max_po_overlap = 0.0
    max_po_time = 0.0
    max_pw_overlap = 0.0
    max_pw_time = 0.0
    
    current_time = None
    frames = {} # time -> list of particles
    
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            
            t = float(parts[0])
            # id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            
            if t not in frames:
                frames[t] = []
            frames[t].append((x, y))
            
    for t, particles in frames.items():
        # Particle-particle
        for i in range(len(particles)):
            p1 = particles[i]
            x1, y1 = p1
            dist_origin = math.sqrt(x1**2 + y1**2)
            
            # Particle-obstacle: max(0, r+r0-dist_to_origin)
            po = r + r0 - dist_origin
            if po > max_po_overlap:
                max_po_overlap = po
                max_po_time = t
            
            # Particle-wall: max(0, r+dist_to_origin-L/2)
            pw = r + dist_origin - (L / 2.0)
            if pw > max_pw_overlap:
                max_pw_overlap = pw
                max_pw_time = t
                
            for j in range(i + 1, len(particles)):
                p2 = particles[j]
                dx = x1 - p2[0]
                dy = y1 - p2[1]
                dist = math.sqrt(dx*dx + dy*dy)
                pp = (2 * r) - dist
                if pp > max_pp_overlap:
                    max_pp_overlap = pp
                    max_pp_time = t
                    
    print(f"Max PP Overlap: {max_pp_overlap} at t={max_pp_time}")
    print(f"Max PO Overlap: {max_po_overlap} at t={max_po_time}")
    print(f"Max PW Overlap: {max_pw_overlap} at t={max_pw_time}")

compute()
