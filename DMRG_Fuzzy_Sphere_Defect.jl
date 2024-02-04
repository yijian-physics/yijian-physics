using ITensors,LinearAlgebra,WignerSymbols

function get_V_from_M(N,i,j,k,l,V0,V1,M0,M1)
    ## N = 2s+1, get V[i,j,k,l]
    s = (N-1)/2
    m1 = -s+i-1
    m2 = -s+j-1
    m3 = -s+k-1
    m4 = -s+l-1
    if(abs(m1+m2)<=N-1 && abs(m3+m4)<=N-1)
        tmp1 = V0 * (4*s+1) * M0[i,j] *  M0[l,k]
    else
        tmp1 = 0
    end
    if(abs(m1+m2)<=N-2 && abs(m3+m4)<=N-2)
        tmp2 = V1 * (4*s-1) * M1[i,j] *  M1[l,k]
    else
        tmp2 = 0
    end
    V = tmp1+tmp2
    return V
end

function compute_all_3js(N)
    M0 = zeros(N,N)
    M1 = zeros(N,N)
    s = (N-1)/2
    for i in 1:N
        for j in 1:N
            m1 = -s+i-1
            m2 = -s+j-1
            if(abs(m1+m2)<=N-1)
                M0[i,j] = wigner3j(s,s,2*s,m1,m2,-m1-m2)
            end
            if(abs(m1+m2)<=N-2)
                M1[i,j] = wigner3j(s,s,2*s-1,m1,m2,-m1-m2)
            end
        end
    end
    return M0,M1
end

function ITensors.space(
  ::SiteType"Electron";
  Lz::Int = 1,
  conserve_sz=false,
  conserve_nf=true,
  conserve_lz=true,
  conserve_spin_parity=true,
  qnname_sz="Sz",
  qnname_nf="Nf",
  qnname_lz="Lz",
  qnname_spin_parity="Z2"
)
  ## Note that conserve nf and lz is always set on. Lz is TWICE the actual angular momentum
  if ((!conserve_sz) && conserve_spin_parity)
    return [
      QN((qnname_nf, 0, -1), (qnname_lz, 0), (qnname_spin_parity, 0, 2)) => 1
      QN((qnname_nf, 1, -1), (qnname_lz, Lz),(qnname_spin_parity, 1, 2)) => 1
      QN((qnname_nf, 1, -1), (qnname_lz, Lz),(qnname_spin_parity, 0, 2)) => 1
      QN((qnname_nf, 2, -1), (qnname_lz, 2*Lz),(qnname_spin_parity, 1, 2)) => 1
    ]
  elseif (!conserve_sz)
    return [
      QN((qnname_nf, 0, -1), (qnname_lz, 0)) => 1
      QN((qnname_nf, 1, -1), (qnname_lz, Lz)) => 1
      QN((qnname_nf, 1, -1), (qnname_lz, Lz)) => 1
      QN((qnname_nf, 2, -1), (qnname_lz, 2*Lz)) => 1
    ]        
  end
  return 4
end

mutable struct myObserver <: AbstractObserver
   energy_tol::Float64
   last_energy::Float64

   myObserver(energy_tol=0.0) = new(energy_tol,1000.0)
end

function ITensors.checkdone!(o::myObserver;kwargs...)
  sw = kwargs[:sweep]
  energy = kwargs[:energy]
  if abs(energy-o.last_energy)/abs(energy) < o.energy_tol
    println("Stopping DMRG after sweep $sw")
    return true
  end
  # Otherwise, update last_energy and keep going
  o.last_energy = energy
  return false
end

function DMRG_Fuzzy_Sphere_Ising_defect(N::Int,hN,hS;V0=4.75,V1=1.0,h=3.16,hz=1000.0,max_bond_dim=400,nsweeps=8,epsilon=1E-10,etol=1E-12)
    ## ZZ+X convention, does not conserve Z2
    ## hN, hS = "+","0","-"
    M0,M1 = compute_all_3js(N)
    sites = [siteind("Electron",Lz=2*i-N-1, conserve_sz=false,conserve_nf=true,conserve_lz=true,conserve_spin_parity=false) for i in 1:N]
    os = OpSum()
    for i in 1:N
        for j in 1:N
            for k in 1:N
                for l in 1:N
                    if(i+j == k+l)
                        V = get_V_from_M(N,i,j,k,l,V0,V1,M0,M1)
                        os += 2*V,"Cdagup",i,"Cup",l,"Cdagdn",j,"Cdn",k
                    end
                end
            end
        end
    end
    
    for i in 1:N
        os += -2*h, "Sx", i
    end
    
    if(hN=="+")
        os += -hz, "Sz", 1
    elseif(hN=="-")
        os += hz, "Sz", 1
    end
    
    if(hS=="+")
        os += -hz, "Sz", N
    elseif(hS=="-")
        os += hz, "Sz", N
    end
    
    H = MPO(os,sites,splitblocks=false)
    
    maxdim = [10,20,100,200,max_bond_dim] 
    noise = [1E-4,1E-6,1E-8,0,0] #add some noise
    cutoff = [epsilon] 
    
    state = ["Up" for n=1:N]
    if(hN=="-")
        state[1] = "Dn"
    end
    if(hS=="-")
        state[N] = "Dn"
    end
    
    psi_prod = MPS(sites,state)
   
    obs = myObserver(etol)
    
    E0,psi0 = dmrg(H,psi_prod; nsweeps, maxdim, cutoff,noise,observer=obs);  # ground state 
    return E0,psi0
end

function DMRG_Fuzzy_Sphere_Ising_10(N::Int;V0=4.75,V1=1.0,h=3.16,max_bond_dim=400,nsweeps=8,epsilon=1E-10,etol=1E-12)
    ## ZZ+X convention, does not conserve Z2
    ## pin the first orbitial to "Up"
    M0,M1 = compute_all_3js(N)
    sites = [siteind("Electron",Lz=2*i-N-1, conserve_sz=false,conserve_nf=true,conserve_lz=true,conserve_spin_parity=false) for i in 1:N]
    sites_trunc = sites[2:N] #pin the first orbital
    os = OpSum()
    for i in 2:N
        for j in 2:N
            for k in 2:N
                for l in 2:N
                    if(i+j == k+l)
                        V = get_V_from_M(N,i,j,k,l,V0,V1,M0,M1)
                        os += 2*V,"Cdagup",i-1,"Cup",l-1,"Cdagdn",j-1,"Cdn",k-1
                    end
                end
            end
        end
    end
    
    for j in 2:N
        V = get_V_from_M(N,1,j,j,1,V0,V1,M0,M1)
        os += 2*V,"Cdagdn",j-1,"Cdn",j-1
    end
    
    for i in 2:N
        os += -2*h, "Sx", i-1
    end
    
    H = MPO(os,sites_trunc,splitblocks=false)
    
    maxdim = [10,20,100,200,max_bond_dim] 
    noise = [1E-4,1E-6,1E-8,0,0] #add some noise
    cutoff = [epsilon] 
    
    state = ["Up" for n=2:N]
    
    psi_prod = MPS(sites_trunc,state)
   
    obs = myObserver(etol)
    
    E0,psi0 = dmrg(H,psi_prod; nsweeps, maxdim, cutoff,noise,observer=obs);  # ground state
    
    return E0,psi0
end

function DMRG_Fuzzy_Sphere_Ising_11(N::Int;V0=4.75,V1=1.0,h=3.16,max_bond_dim=400,nsweeps=8,epsilon=1E-10,etol=1E-12)
    ## ZZ+X convention, does not conserve Z2
    ## pin the first and last orbitial to "Up"
    M0,M1 = compute_all_3js(N)
    sites = [siteind("Electron",Lz=2*i-N-1, conserve_sz=false,conserve_nf=true,conserve_lz=true,conserve_spin_parity=false) for i in 1:N]
    sites_trunc = sites[2:N-1] #pin the first orbital, pin last orbital
    os = OpSum()
    for i in 2:N-1
        for j in 2:N-1
            for k in 2:N-1
                for l in 2:N-1
                    if(i+j == k+l)
                        V = get_V_from_M(N,i,j,k,l,V0,V1,M0,M1)
                        os += 2*V,"Cdagup",i-1,"Cup",l-1,"Cdagdn",j-1,"Cdn",k-1
                    end
                end
            end
        end
    end
    
    for j in 2:N-1
        V_first = get_V_from_M(N,1,j,j,1,V0,V1,M0,M1) # projection of the first orbital
        V_last = get_V_from_M(N,j,N,N,j,V0,V1,M0,M1) # projection of the last orbital
        os += 2*(V_first+V_last),"Cdagdn",j-1,"Cdn",j-1
    end
    
    for i in 2:N-1
        os += -2*h, "Sx", i-1
    end
    
    H = MPO(os,sites_trunc,splitblocks=false)
    
    maxdim = [10,20,100,200,max_bond_dim] 
    noise = [1E-4,1E-6,1E-8,0,0] #add some noise
    cutoff = [epsilon] 
    
    state = ["Up" for n=2:N-1]
    
    psi_prod = MPS(sites_trunc,state)
   
    obs = myObserver(etol)
    
    E0,psi0 = dmrg(H,psi_prod; nsweeps, maxdim, cutoff,noise,observer=obs);  # ground state
    
    return E0,psi0
end

function compute_g(N::Int;max_bd=2000,max_sweep=15)
    ~,psi00 = DMRG_Fuzzy_Sphere_Ising_defect(N,"0","0",max_bond_dim = max_bd,nsweeps=max_sweep);
    ~,psi10 = DMRG_Fuzzy_Sphere_Ising_defect(N,"+","0",max_bond_dim = max_bd,nsweeps=max_sweep);
    ~,psi11 = DMRG_Fuzzy_Sphere_Ising_defect(N,"+","+",max_bond_dim = max_bd,nsweeps=max_sweep);
    g = (inner(psi00',psi10)/inner(psi10',psi11))^2
    return g
end

function compute_g_pinned(N::Int;max_bd=2000,max_sweep=15)
    ~,psi00 = DMRG_Fuzzy_Sphere_Ising_defect(N,"0","0",max_bond_dim = max_bd,nsweeps=max_sweep);
    ~,psi10 = DMRG_Fuzzy_Sphere_Ising_10(N,max_bond_dim = max_bd,nsweeps=max_sweep);
    ~,psi11 = DMRG_Fuzzy_Sphere_Ising_11(N,max_bond_dim = max_bd,nsweeps=max_sweep);
    
    first_up_state = dag(state(inds(psi00[1])[2],"Up"))
    proj_first_tensor = contract(psi00[1],first_up_state)
    new_first_tensor = contract(proj_first_tensor,psi00[2]);
    psi00_proj = MPS(vcat(new_first_tensor,psi00[3:end]))
    
    last_up_state = dag(state(inds(psi10[end])[1],"Up"))
    proj_last_tensor = contract(psi10[end],last_up_state)
    new_last_tensor = contract(psi10[end-1],proj_last_tensor);
    psi10_proj = MPS(vcat(psi10[1:end-2],new_last_tensor))
    
    g = (inner(psi00_proj',psi10)/inner(psi10_proj',psi11))^2
    return g
end

N = 12
g = compute_g_pinned(N)
@show g