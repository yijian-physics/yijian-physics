using ITensors,LinearAlgebra

using WignerSymbols

ITensors.enable_threaded_blocksparse(true) 
ITensors.Strided.disable_threads() #Using block sparse parallel computation instead of BLAS multithreading

function get_V(N,i,j,k,l,V0,V1)
    ## N = 2s+1, get V[i,j,k,l]
    s = (N-1)/2
    m1 = -s+i-1
    m2 = -s+j-1
    m3 = -s+k-1
    m4 = -s+l-1
    if(abs(m1+m2)<=N-1 && abs(m3+m4)<=N-1)
        tmp1 = V0 * (4*s+1) * wigner3j(s,s,2*s,m1,m2,-m1-m2) *  wigner3j(s,s,2*s,m4,m3,-m3-m4)
    else
        tmp1 = 0
    end
    if(abs(m1+m2)<=N-2 && abs(m3+m4)<=N-2)
        tmp2 = V1 * (4*s-1) * wigner3j(s,s,2*s-1,m1,m2,-m1-m2) *  wigner3j(s,s,2*s-1,m4,m3,-m3-m4)
    else
        tmp2 = 0
    end
    V = tmp1+tmp2
    return V
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

function DMRG_Fuzzy_Sphere_Ising_conserveZ2(N::Int;V0=4.75,V1=1.0,h=3.16,max_bond_dim=400,nsweeps=8,epsilon=1E-10,etol=1E-12)
    ## Switch to XX+Z convention
    sites = [siteind("Electron",Lz=2*i-N-1, conserve_sz=false,conserve_nf=true,conserve_lz=true,conserve_spin_parity=true) for i in 1:N]
    os = OpSum()
    for i in 1:N
        for j in 1:N
            for k in 1:N
                for l in 1:N
                    if(i+j == k+l)
                        V = get_V(N,i,j,k,l,V0,V1)
                        os += V/2,"Cdagup",i,"Cup",l,"Cdagup",j,"Cup",k
                        os += V/2,"Cdagdn",i,"Cdn",l,"Cdagdn",j,"Cdn",k
                        os += V/2,"Cdagup",i,"Cup",l,"Cdagdn",j,"Cdn",k
                        os += V/2,"Cdagdn",i,"Cdn",l,"Cdagup",j,"Cup",k # H00
                        
                        os += -V/2,"Cdagup",i,"Cdn",l,"Cdagup",j,"Cdn",k
                        os += -V/2,"Cdagup",i,"Cdn",l,"Cdagdn",j,"Cup",k
                        os += -V/2,"Cdagdn",i,"Cup",l,"Cdagup",j,"Cdn",k
                        os += -V/2,"Cdagdn",i,"Cup",l,"Cdagdn",j,"Cup",k # Hxx
                    end
                end
            end
        end
    end
    
    for i in 1:N
        os += -2*h, "Sz", i
    end
    H = MPO(os,sites,splitblocks=false)
    
    maxdim = [10,20,100,200,max_bond_dim] 
    noise = [1E-4,1E-6,1E-8,0,0] #add some noise
    cutoff = [epsilon] 
    
    state_even = ["Up" for n=1:N]
    psi_prod_even = MPS(sites,state_even)
    
    state_odd = ["Up" for n=1:N]
    state_odd[div(N,2)] = "Dn"
    psi_prod_odd = MPS(sites,state_odd)
   
    obs = myObserver(etol)
    
    E0,psi0 = dmrg(H,psi_prod_even; nsweeps, maxdim, cutoff,noise,observer=obs);  # ground state 
    E2,psi2 = dmrg(H,[psi0],psi_prod_even; nsweeps, maxdim, cutoff,noise,weight=N,observer=obs);  # epsilon state
    
    E1,psi1 = dmrg(H,psi_prod_odd; nsweeps, maxdim, cutoff,noise,observer=obs);  # sigma state
    return [E0,E1,E2],[psi0,psi1,psi2]
end


function DMRG_Fuzzy_Sphere_Ising_no_conserveZ2(N::Int;V0=4.75,V1=1.0,h=3.16,max_bond_dim=400,nsweeps=8,epsilon=1E-10,etol=1E-12)
    ## ZZ+X convention, does not conserve Z2
    sites = [siteind("Electron",Lz=2*i-N-1, conserve_sz=false,conserve_nf=true,conserve_lz=true,conserve_spin_parity=false) for i in 1:N]
    os = OpSum()
    for i in 1:N
        for j in 1:N
            for k in 1:N
                for l in 1:N
                    if(i+j == k+l)
                        V = get_V(N,i,j,k,l,V0,V1)
                        os += 2*V,"Cdagup",i,"Cup",l,"Cdagdn",j,"Cdn",k
                    end
                end
            end
        end
    end
    
    for i in 1:N
        os += -2*h, "Sx", i
    end
    H = MPO(os,sites,splitblocks=false)
    
    maxdim = [10,20,100,200,max_bond_dim] 
    noise = [1E-4,1E-6,1E-8,0,0] #add some noise
    cutoff = [epsilon] 
    
    state = ["Up" for n=1:N]
    psi_prod = MPS(sites,state)
   
    obs = myObserver(etol)
    
    E0,psi0 = dmrg(H,psi_prod; nsweeps, maxdim, cutoff,noise,observer=obs);  # ground state 
    E1,psi1 = dmrg(H,[psi0],psi_prod; nsweeps, maxdim, cutoff,noise,weight=N,observer=obs);  # sigma state
    E2,psi2 = dmrg(H,[psi0,psi1],psi_prod; nsweeps, maxdim, cutoff,noise,weight=N,observer=obs);  # epsilon state
    return [E0,E1,E2],[psi0,psi1,psi2]
end
    
let
    N = 10
    Es,psis = DMRG_Fuzzy_Sphere_Ising_no_conserveZ2(N);
    @show Es
    Es,psis = DMRG_Fuzzy_Sphere_Ising_conserveZ2(N);
    @show Es
end