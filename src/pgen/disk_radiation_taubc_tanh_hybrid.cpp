//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file disk.cpp
//! \brief Initializes stratified Keplerian accretion disk in both cylindrical and
//! spherical polar coordinates.  Initial conditions are in vertical hydrostatic eqm.

// C headers

// C++ headers
#include <algorithm>  // min
#include <cmath>      // sqrt
#include <cstdlib>    // srand
#include <cstring>    // strcmp()
#include <fstream>
#include <iostream>   // endl
#include <limits>
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/bvals.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../orbital_advection/orbital_advection.hpp"
#include "../parameter_input.hpp"

#include "../radiation/radiation.hpp"
#include "../hydro/srcterms/hydro_srcterms.hpp"
#include "../radiation/integrators/rad_integrators.hpp"
#include "../utils/utils.hpp"


namespace {
void GetCylCoord(Coordinates *pco,Real &rad,Real &phi,Real &z,int i,int j,int k);
Real DenProfileCyl(const Real rad, const Real phi, const Real z);
Real PoverR(const Real rad, const Real phi, const Real z);
Real VelProfileCyl(const Real rad, const Real phi, const Real z);
// problem parameters which are useful to make global to this file
Real gm0, r0, rho0, dslope, p0_over_r0, pslope, gamma_gas;
Real dfloor;
Real Omega0;
Real kappa_s, kappa_a, kappa_planck, kappa_star;
Real lunit, rhounit, tunit; //
Real tfloor, tceiling; //
Real t_star, r_star, tau0, t_bkg, x1min, cavity_fac, r_c, dgratio, small_grain_ratio, width_tanh, sigma_shadow, amp_shadow, t_growth, prat, crat;

Real h_planck_cgs = 6.6260755e-27; // Planck constant
Real k_b_cgs = 1.380649e-16;   // Boltzman constant
Real arad_cgs = 7.565733250033928e-15;
Real c_speed_cgs = 2.99792458e10;
Real G_cgs = 6.6743e-8; //
Real Msun_cgs = 1.989e33; //
int len_freq;

static AthenaArray<Real> column_rho_bc;
static AthenaArray<Real> stellar_opacity_freq;
static AthenaArray<Real> stellar_opacity;
static AthenaArray<Real> planck_function;

void Inner_rad_X1(MeshBlock *pmb, Coordinates *pco, Radiation *prad,
     const AthenaArray<Real> &w, FaceField &b, AthenaArray<Real> &ir,
      Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
void Outer_rad_X1(MeshBlock *pmb, Coordinates *pco, Radiation *prad,
     const AthenaArray<Real> &w, FaceField &b, AthenaArray<Real> &ir,
      Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
void Inner_rad_X2(MeshBlock *pmb, Coordinates *pco, Radiation *prad,
     const AthenaArray<Real> &w, FaceField &b, AthenaArray<Real> &ir,
      Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
void Outer_rad_X2(MeshBlock *pmb, Coordinates *pco, Radiation *prad,
     const AthenaArray<Real> &w, FaceField &b, AthenaArray<Real> &ir,
      Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);

void DiskOpacity(MeshBlock *pmb, AthenaArray<Real> &prim);

//! stellar heating Flock et al. 2020
void StellarHeating(MeshBlock *pmb, const Real time, const Real dt,
             const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
             const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
             AthenaArray<Real> &cons_scalar);

//! fifth polynomial fit of planck opacity as function of temperature
//! Inputs and outputs are in physical units
void PolyOpacityPlanck(const Real tgas, Real &kappa);

void PolyOpacityRosseland(const Real tgas, Real &kappa);

void GaussianAttenuation(const Real sintheta, const Real sinphi,
                         const Real amp, const Real sigma, const Real t_growth,
                         Real time, Real &attenuation);

} // namespace

// User-defined boundary conditions for disk simulations
void DiskInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//! \brief Function to initialize problem-specific data in mesh class.  Can also be used
//! to initialize variables which are global to (and therefore can be passed to) other
//! functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // Get parameters for gravitatonal potential of central point mass
  gm0 = pin->GetOrAddReal("problem","GM",0.0);
  r0 = pin->GetOrAddReal("problem","r0",1.0);

  // Get parameters for initial density and velocity
  rho0 = pin->GetReal("problem","rho0");
  dslope = pin->GetOrAddReal("problem","dslope",0.0);

  // Get parameters of initial pressure and cooling parameters
  if (NON_BAROTROPIC_EOS) {
    p0_over_r0 = pin->GetOrAddReal("problem","p0_over_r0",0.0025);
    pslope = pin->GetOrAddReal("problem","pslope",0.0);
    gamma_gas = pin->GetReal("hydro","gamma");
  } else {
    p0_over_r0=SQR(pin->GetReal("hydro","iso_sound_speed"));
  }
  Real float_min = std::numeric_limits<float>::min();
  dfloor=pin->GetOrAddReal("hydro","dfloor",(1024*(float_min)));

  Omega0 = pin->GetOrAddReal("orbital_advection","Omega0",0.0);

  if(RADIATION_ENABLED || IM_RADIATION_ENABLED){
    kappa_s = pin->GetOrAddReal("radiation","kappa_s", 1.);
    kappa_a = pin->GetOrAddReal("radiation","kappa_a", 1.);
    kappa_planck = pin->GetOrAddReal("radiation","kappa_planck", 0.5);
    kappa_star = pin->GetOrAddReal("radiation","kappa_star", 1.);
    
    tfloor = pin->GetOrAddReal("radiation", "tfloor", 0.001);
    tceiling = pin->GetOrAddReal("radiation", "tceiling", 2);
    rhounit = pin->GetOrAddReal("radiation", "rhounit", 1.e-8);
    tunit = pin->GetOrAddReal("radiation", "Tunit", 2.e5);
    lunit = pin->GetOrAddReal("radiation", "lunit", 1.48428e14);
   
    r_star = pin->GetOrAddReal("radiation", "r_star", 1.);
    t_star = pin->GetOrAddReal("radiation", "t_star", 1.);
    tau0   = pin->GetOrAddReal("radiation", "tau0", 0.);
    t_bkg  = pin->GetOrAddReal("radiation", "t_bkg", 0.);
    cavity_fac = pin->GetOrAddReal("radiation", "cavity_fac", 0.);
    r_c    = pin->GetOrAddReal("radiation", "r_c", 1e6);
    width_tanh = pin->GetOrAddReal("radiation", "width_tanh", 1e4);
    
    x1min  = pin->GetOrAddReal("mesh", "x1min", 0.);
    dgratio= pin->GetOrAddReal("radiation", "dgratio", 0.01);
    small_grain_ratio = pin->GetOrAddReal("radiation", "small_grain_ratio", 0.02856);
    kappa_star = kappa_star * dgratio * small_grain_ratio  * rhounit * lunit;
    sigma_shadow = pin->GetOrAddReal("problem", "sigma_shadow", 0.1);
    amp_shadow = pin->GetOrAddReal("problem", "amp_shadow", 0.9);
    t_growth   = pin->GetOrAddReal("problem", "t_growth", 62.8);
    
    prat = pin->GetOrAddReal("radiation", "Prat", 1e4);
    crat = pin->GetOrAddReal("radiation", "Crat", 1e4);
  }
    
    

  // enroll user-defined boundary condition
  if (mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x1, DiskInnerX1);
  }
  if (mesh_bcs[BoundaryFace::outer_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x1, DiskOuterX1);
  }
  if (mesh_bcs[BoundaryFace::inner_x2] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x2, DiskInnerX2);
  }
  if (mesh_bcs[BoundaryFace::outer_x2] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x2, DiskOuterX2);
  }
  if (mesh_bcs[BoundaryFace::inner_x3] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x3, DiskInnerX3);
  }
  if (mesh_bcs[BoundaryFace::outer_x3] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x3, DiskOuterX3);
  }

  if(RADIATION_ENABLED || IM_RADIATION_ENABLED){
    if (mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user")) {
      EnrollUserRadBoundaryFunction(BoundaryFace::inner_x1, Inner_rad_X1);
    }
    if (mesh_bcs[BoundaryFace::outer_x1] == GetBoundaryFlag("user")) {
      EnrollUserRadBoundaryFunction(BoundaryFace::outer_x1, Outer_rad_X1);
    }
    if (mesh_bcs[BoundaryFace::inner_x2] == GetBoundaryFlag("user")) {
      EnrollUserRadBoundaryFunction(BoundaryFace::inner_x2, Inner_rad_X2);
    }
    if (mesh_bcs[BoundaryFace::outer_x2] == GetBoundaryFlag("user")) {
      EnrollUserRadBoundaryFunction(BoundaryFace::outer_x2, Outer_rad_X2);
    }

    EnrollUserExplicitSourceFunction(StellarHeating);
    // same as /radiation/radiation.cpp
    int nfreq = pin->GetOrAddInteger("radiation","n_frequency",1);
    AllocateRealUserMeshDataField(1);
    ruser_mesh_data[0].NewAthenaArray(mesh_size.nx3,
                                      mesh_size.nx2,
                                      nrbx1);
    for (int k=0; k<mesh_size.nx3; k++){
      for (int j=0; j<mesh_size.nx2; j++){
        for (int ib=0; ib<nrbx1; ib++){
          ruser_mesh_data[0](k,j,ib)=0.0;
        }
      }
    }
    // flag to initialize mesh data to zero.
    AllocateIntUserMeshDataField(1);
    iuser_mesh_data[0].NewAthenaArray(1);
    iuser_mesh_data[0](0) = 0;
    
    if(RADIATION_ENABLED || IM_RADIATION_ENABLED){
      if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0){
        // column density inside the inner boundary
        column_rho_bc.NewAthenaArray(mesh_size.nx2);
        // read in the column density at the inner boundary
        FILE *frhobc;
        if ( (frhobc=fopen("./column_rho_bc.txt","r"))==NULL )
        {
          printf("Open boundary radial column density input file error");
          exit(0);
          return;
        }
        for(int i=0; i<mesh_size.nx2; i++){
          fscanf(frhobc,"%lf",&(column_rho_bc(i)));
        }
        fclose(frhobc);
      }
      // read in wavelength and opacity
      FILE *fstellaropac;
      if ( (fstellaropac=fopen("./stellar_opacity.txt","r"))==NULL )
      {
        printf("Open stellar opacity input file error");
        exit(0);
        return;
      }
      fscanf(fstellaropac,"%d",&len_freq);
      stellar_opacity_freq.NewAthenaArray(len_freq);
      stellar_opacity.NewAthenaArray(len_freq);
      planck_function.NewAthenaArray(len_freq);
      
      for(int ifrs=0; ifrs<len_freq; ifrs++){
        fscanf(fstellaropac,"%lf %lf",&stellar_opacity_freq(ifrs),
                                      &stellar_opacity(ifrs));
        //std::cout << stellar_opacity_freq(ifrs) << " " << stellar_opacity(ifrs) << std::endl;
        stellar_opacity(ifrs) = stellar_opacity(ifrs)
        * dgratio * small_grain_ratio  * rhounit * lunit;
        planck_function(ifrs) = 2*h_planck_cgs/c_speed_cgs/c_speed_cgs
        *stellar_opacity_freq(ifrs)*stellar_opacity_freq(ifrs)*stellar_opacity_freq(ifrs)
        /(std::exp(h_planck_cgs*stellar_opacity_freq(ifrs)/(k_b_cgs*t_star*tunit)) - 1.);
      }
      // Real speed_of_light = c_speed_cgs / std::sqrt(gm0*G_cgs*Msun_cgs/lunit);
      // Real conversion_factor = PI / (arad_cgs*c_speed_cgs/4. * std::pow(tunit*t_star, 4)) * prat*speed_of_light/4. *t_star*t_star*t_star*t_star;
      Real conversion_factor = PI / (arad_cgs*c_speed_cgs/4. * std::pow(tunit, 4)) * prat*crat/4.;
      
      // Real total_planck = 0; // for test purpose
      for(int ifrs=0; ifrs<len_freq; ifrs++){
        Real dlnfreq;
        if (ifrs==0){
          dlnfreq = std::log(stellar_opacity_freq(1)) - std::log(stellar_opacity_freq(0));
        }else{
          dlnfreq = std::log(stellar_opacity_freq(ifrs)) - std::log(stellar_opacity_freq(ifrs-1));
        }
        planck_function(ifrs) = planck_function(ifrs) * stellar_opacity_freq(ifrs) * dlnfreq * conversion_factor;
        // total_planck += planck_function(ifrs);
        //std::cout << stellar_opacity(ifrs) << std::endl;
        //std::cout << kappa_star << std::endl;
                      
      }
      fclose(fstellaropac);
      //std::cout << total_planck << std::endl;
    }
    
  }
  return;
}

void Mesh::UserWorkAfterLoop(ParameterInput *pin)
{
  // free memory
  if(RADIATION_ENABLED || IM_RADIATION_ENABLED){
    if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0){
      column_rho_bc.DeleteAthenaArray();
    }
  }
  return;
}

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin)
{

  
  if(RADIATION_ENABLED || IM_RADIATION_ENABLED){
    prad->EnrollOpacityFunction(DiskOpacity);
    // initialize tau in meshblock
    AllocateRealUserMeshBlockDataField(1);
    ruser_meshblock_data[0].NewAthenaArray(block_size.nx3,
                                           block_size.nx2,
                                           block_size.nx1);
    for (int k=0; k<block_size.nx3; k++){
      for (int j=0; j<block_size.nx2; j++){
        for (int i=0; i<block_size.nx1; i++){
          ruser_meshblock_data[0](k,j,i) = 0.0;
        }
      }
    }
  }
  return;
}


//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Initializes Keplerian accretion disk.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Real rad(0.0), phi(0.0), z(0.0);
  Real den, vel;
  Real x1, x2, x3;
    
  AthenaArray<Real> ir_cm;
  Real *ir_lab;
  Real crat, prat;

  if(RADIATION_ENABLED || IM_RADIATION_ENABLED){
    ir_cm.NewAthenaArray(prad->n_fre_ang);
    crat = prad->crat;
    prat = prad->prat;
  }else{
    crat = 5694.76;
    prat = 0.0;
    }


  


  OrbitalVelocityFunc &vK = porb->OrbitalVelocity;
  //  Initialize density and momenta
  for (int k=ks; k<=ke; ++k) {
    x3 = pcoord->x3v(k);
    for (int j=js; j<=je; ++j) {
      x2 = pcoord->x2v(j);
      for (int i=is; i<=ie; ++i) {
        x1 = pcoord->x1v(i);
        GetCylCoord(pcoord,rad,phi,z,i,j,k); // convert to cylindrical coordinates
        // compute initial conditions in cylindrical coordinates
        den = DenProfileCyl(rad,phi,z);
        vel = VelProfileCyl(rad,phi,z);
        if (porb->orbital_advection_defined)
          vel -= vK(porb, x1, x2, x3);
        phydro->u(IDN,k,j,i) = den;
        phydro->u(IM1,k,j,i) = 0.0;
        if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
          phydro->u(IM2,k,j,i) = den*vel;
          phydro->u(IM3,k,j,i) = 0.0;
        } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
          phydro->u(IM2,k,j,i) = 0.0;
          phydro->u(IM3,k,j,i) = den*vel;
        }

        if (NON_BAROTROPIC_EOS) {
          Real p_over_r = PoverR(rad,phi,z);
          phydro->u(IEN,k,j,i) = p_over_r*phydro->u(IDN,k,j,i)/(gamma_gas - 1.0);
          phydro->u(IEN,k,j,i) += 0.5*(SQR(phydro->u(IM1,k,j,i))+SQR(phydro->u(IM2,k,j,i))
                                       + SQR(phydro->u(IM3,k,j,i)))/phydro->u(IDN,k,j,i);
        }
          
        /* if(RADIATION_ENABLED || IM_RADIATION_ENABLED){
          Real gast = PoverR(rad,phi,z);
          for(int n=0; n<prad->n_fre_ang; ++n)
            ir_cm(n) = gast * gast * gast * gast;

          Real *mux = &(prad->mu(0,k,j,i,0));
          Real *muy = &(prad->mu(1,k,j,i,0));
          Real *muz = &(prad->mu(2,k,j,i,0));

          ir_lab = &(prad->ir(k,j,i,0));
          prad->pradintegrator->ComToLab(0,0,vel,mux,muy,muz,ir_cm,ir_lab);
   
        }// End Rad*/

      }
    }
  }

  return;
}

void MeshBlock::UserWorkInLoop(void){
  if(RADIATION_ENABLED || IM_RADIATION_ENABLED){
    int lx3 = static_cast<int>(loc.lx3);
    int lx2 = static_cast<int>(loc.lx2);
    int lx1 = static_cast<int>(loc.lx1);
    int nrbx1 = pmy_mesh->mesh_size.nx1/block_size.nx1;
    // step 1. initialize ruser_mesh_data
    if (pmy_mesh->iuser_mesh_data[0](0) == 0){
      for (int k=0; k<pmy_mesh->mesh_size.nx3; k++){
        for (int j=0; j<pmy_mesh->mesh_size.nx2; j++){
          for (int ib=0; ib<nrbx1; ib++){
            pmy_mesh->ruser_mesh_data[0](k,j,ib)=0.0;
          }
        }
      }
      pmy_mesh->iuser_mesh_data[0](0) = 1;
    }
    // step 2.
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        // initialize the leftmost x1 to be zero
        ruser_meshblock_data[0](k-ks,j-js,0) =
        kappa_star*phydro->u(IDN,k,j,is)*pcoord->dx1f(is);
        for (int i = is+1; i <= ie; ++i) {
          Real dtau = kappa_star
                         *phydro->u(IDN,k,j,i)*pcoord->dx1f(i);
          ruser_meshblock_data[0](k-ks,j-js,i-is) =
          ruser_meshblock_data[0](k-ks,j-js,i-1-is) + dtau;
        }
      }
    }
    
    
    // step 3. the first radial location is tau0, else lx1+1 is the dtau in lx1 block. The last lx1 does not save.
    if (lx1 != nrbx1-1) {
      for (int k=ks; k<=ke; k++) {
        for (int j=js; j<=je; j++) {
          int tj  = lx2*block_size.nx2+(j-js);
          int tk  = lx3*block_size.nx3+(k-ks);
          if (lx1 == 0){
            // pmy_mesh->ruser_mesh_data[0](tk,tj,0) = tau0;
            // is or is-1?
            pmy_mesh->ruser_mesh_data[0](tk,tj,0) = 0.; // kappa_star*column_rho_bc(tj);
          }
          pmy_mesh->ruser_mesh_data[0](tk,tj,lx1+1) =
          ruser_meshblock_data[0](k-ks,j-js,ie-is);
        }
      }

    }else if (nrbx1 == 1) {
      for (int k=ks; k<=ke; k++) {
        for (int j=js; j<=je; j++) {
          int tj  = lx2*block_size.nx2+(j-js);
          int tk  = lx3*block_size.nx3+(k-ks);
          pmy_mesh->ruser_mesh_data[0](tk,tj,0) = 0.; // kappa_star*column_rho_bc(tj);
        }
      }
    }
  }
  return;
}
      
void Mesh::UserWorkInLoop(void){
  // step 1. integrate the rightmost
  if (RADIATION_ENABLED || IM_RADIATION_ENABLED){
    MeshBlock *pmb = my_blocks(0);
    for (int tk=0; tk<mesh_size.nx3; tk++){
      for (int tj=0; tj<mesh_size.nx2; tj++){
        for (int ib=1; ib<nrbx1; ib++){
          ruser_mesh_data[0](tk,tj,ib) += ruser_mesh_data[0](tk,tj,ib-1);
        }
      }
    }
    // step 2. all reduce
  // allreduce each ruser_mesh_data[0]
  //As this function is called after MeshBlock::UserWorkInLoop, this function can be used to collect the results from MeshBlocks. Also, since this is called only once per node, MPI all-to-all communications can be used here.
#ifdef MPI_PARALLEL
    int ntot = mesh_size.nx3*mesh_size.nx2*nrbx1;
    MPI_Allreduce(MPI_IN_PLACE,
                  ruser_mesh_data[0].data(),
                  ntot, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
    // std::cout << "Mesh, my_rank " << Globals::my_rank << ' '
    // << ruser_mesh_data[0](0,32,0,0) << ' '
    // << ruser_mesh_data[0](0,32,1,0) << std::endl;
    // step 3. set the flag to zero.
    iuser_mesh_data[0](0) = 0;
  }
  return;
}
      
namespace {
//----------------------------------------------------------------------------------------
//! transform to cylindrical coordinate

void GetCylCoord(Coordinates *pco,Real &rad,Real &phi,Real &z,int i,int j,int k) {
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    rad=pco->x1v(i);
    phi=pco->x2v(j);
    z=pco->x3v(k);
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    rad=std::abs(pco->x1v(i)*std::sin(pco->x2v(j)));
    phi=pco->x3v(k);
    z=pco->x1v(i)*std::cos(pco->x2v(j));
  }
  return;
}

//----------------------------------------------------------------------------------------
//! computes density in cylindrical coordinates
//! carves a gap to separate the inner disk and the inner boundary
/*Real DenProfileCyl(const Real rad, const Real phi, const Real z) {
  Real den;
  Real p_over_r = p0_over_r0;
  if (NON_BAROTROPIC_EOS) p_over_r = PoverR(rad, phi, z);
  Real denmid = rho0*std::pow(rad/r0,dslope)*std::exp(-rad/r0/r_c);
  Real dentem = denmid*std::exp(gm0/p_over_r*(1./std::sqrt(SQR(rad)+SQR(z))-1./rad));
  Real r1min_2 = x1min*cavity_fac;
  dentem = dentem * (std::tanh((rad-r1min_2)/width_tanh)/2 + 0.5);
  den = dentem;
  return std::max(den,dfloor);
}
*/

Real DenProfileCyl(const Real rad, const Real phi, const Real z) {
  Real den;
  Real p_over_r = p0_over_r0;
  Real EULER = 2.7182818284590452;

  if (NON_BAROTROPIC_EOS) p_over_r = PoverR(rad, phi, z);
  Real denmid = rho0*std::pow((rad + r0)/r0, dslope)\
                /(1 + std::exp(-std::exp(EULER)*(rad - r0)/r0));
  Real dentem = denmid*std::exp(gm0/p_over_r*(1./std::sqrt(SQR(rad)+SQR(z))-1./rad));
  den = dentem;

  return std::max(den,dfloor);
}

//----------------------------------------------------------------------------------------
//! computes pressure/density in cylindrical coordinates

Real PoverR(const Real rad, const Real phi, const Real z) {
  Real poverr;
  poverr = p0_over_r0*std::pow(rad/r0, pslope);
  return poverr;
}

//----------------------------------------------------------------------------------------
//! computes rotational velocity in cylindrical coordinates

Real VelProfileCyl(const Real rad, const Real phi, const Real z) {
  Real p_over_r = PoverR(rad, phi, z);
  Real vel = (dslope+pslope)*p_over_r/(gm0/rad) + (1.0+pslope)
             - pslope*rad/std::sqrt(rad*rad+z*z);
  vel = std::sqrt(gm0/rad)*std::sqrt(vel) - rad*Omega0;
  return vel;
}
} // namespace

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values

void DiskInnerX1(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad(0.0), phi(0.0), z(0.0);
  Real vel;
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=1; i<=ngh; ++i) {
          GetCylCoord(pco,rad,phi,z,il-i,j,k);
          prim(IDN,k,j,il-i) = DenProfileCyl(rad,phi,z);
          vel = VelProfileCyl(rad,phi,z);
          if (pmb->porb->orbital_advection_defined)
            vel -= vK(pmb->porb, pco->x1v(il-i), pco->x2v(j), pco->x3v(k));
          prim(IM1,k,j,il-i) = 0.0;
          prim(IM2,k,j,il-i) = vel;
          prim(IM3,k,j,il-i) = 0.0;
          if (NON_BAROTROPIC_EOS)
            prim(IEN,k,j,il-i) = PoverR(rad, phi, z)*prim(IDN,k,j,il-i);
        }
      }
    }
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=1; i<=ngh; ++i) {
          GetCylCoord(pco,rad,phi,z,il-i,j,k);
          prim(IDN,k,j,il-i) = DenProfileCyl(rad,phi,z);
          vel = VelProfileCyl(rad,phi,z);
          if (pmb->porb->orbital_advection_defined)
            vel -= vK(pmb->porb, pco->x1v(il-i), pco->x2v(j), pco->x3v(k));
          prim(IM1,k,j,il-i) = 0.0;
          prim(IM2,k,j,il-i) = 0.0;
          prim(IM3,k,j,il-i) = vel;
          if (NON_BAROTROPIC_EOS)
            prim(IEN,k,j,il-i) = PoverR(rad, phi, z)*prim(IDN,k,j,il-i);
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values

void DiskOuterX1(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad(0.0), phi(0.0), z(0.0);
  Real vel;
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=1; i<=ngh; ++i) {
          GetCylCoord(pco,rad,phi,z,iu+i,j,k);
          prim(IDN,k,j,iu+i) = DenProfileCyl(rad,phi,z);
          vel = VelProfileCyl(rad,phi,z);
          if (pmb->porb->orbital_advection_defined)
            vel -= vK(pmb->porb, pco->x1v(iu+i), pco->x2v(j), pco->x3v(k));
          prim(IM1,k,j,iu+i) = 0.0;
          prim(IM2,k,j,iu+i) = vel;
          prim(IM3,k,j,iu+i) = 0.0;
          if (NON_BAROTROPIC_EOS)
            prim(IEN,k,j,iu+i) = PoverR(rad, phi, z)*prim(IDN,k,j,iu+i);
        }
      }
    }
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=1; i<=ngh; ++i) {
          GetCylCoord(pco,rad,phi,z,iu+i,j,k);
          prim(IDN,k,j,iu+i) = DenProfileCyl(rad,phi,z);
          vel = VelProfileCyl(rad,phi,z);
          if (pmb->porb->orbital_advection_defined)
            vel -= vK(pmb->porb, pco->x1v(iu+i), pco->x2v(j), pco->x3v(k));
          prim(IM1,k,j,iu+i) = 0.0;
          prim(IM2,k,j,iu+i) = 0.0;
          prim(IM3,k,j,iu+i) = vel;
          if (NON_BAROTROPIC_EOS)
            prim(IEN,k,j,iu+i) = PoverR(rad, phi, z)*prim(IDN,k,j,iu+i);
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values

void DiskInnerX2(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad(0.0), phi(0.0), z(0.0);
  Real vel;
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
        for (int i=il; i<=iu; ++i) {
          GetCylCoord(pco,rad,phi,z,i,jl-j,k);
          prim(IDN,k,jl-j,i) = DenProfileCyl(rad,phi,z);
          vel = VelProfileCyl(rad,phi,z);
          if (pmb->porb->orbital_advection_defined)
            vel -= vK(pmb->porb, pco->x1v(i), pco->x2v(jl-j), pco->x3v(k));
          prim(IM1,k,jl-j,i) = 0.0;
          prim(IM2,k,jl-j,i) = vel;
          prim(IM3,k,jl-j,i) = 0.0;
          if (NON_BAROTROPIC_EOS)
            prim(IEN,k,jl-j,i) = PoverR(rad, phi, z)*prim(IDN,k,jl-j,i);
        }
      }
    }
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
        for (int i=il; i<=iu; ++i) {
          GetCylCoord(pco,rad,phi,z,i,jl-j,k);
          prim(IDN,k,jl-j,i) = DenProfileCyl(rad,phi,z);
          vel = VelProfileCyl(rad,phi,z);
          if (pmb->porb->orbital_advection_defined)
            vel -= vK(pmb->porb, pco->x1v(i), pco->x2v(jl-j), pco->x3v(k));
          prim(IM1,k,jl-j,i) = 0.0;
          prim(IM2,k,jl-j,i) = 0.0;
          prim(IM3,k,jl-j,i) = vel;
          if (NON_BAROTROPIC_EOS)
            prim(IEN,k,jl-j,i) = PoverR(rad, phi, z)*prim(IDN,k,jl-j,i);
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values

void DiskOuterX2(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad(0.0), phi(0.0), z(0.0);
  Real vel;
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
        for (int i=il; i<=iu; ++i) {
          GetCylCoord(pco,rad,phi,z,i,ju+j,k);
          prim(IDN,k,ju+j,i) = DenProfileCyl(rad,phi,z);
          vel = VelProfileCyl(rad,phi,z);
          if (pmb->porb->orbital_advection_defined)
            vel -= vK(pmb->porb, pco->x1v(i), pco->x2v(ju+j), pco->x3v(k));
          prim(IM1,k,ju+j,i) = 0.0;
          prim(IM2,k,ju+j,i) = vel;
          prim(IM3,k,ju+j,i) = 0.0;
          if (NON_BAROTROPIC_EOS)
            prim(IEN,k,ju+j,i) = PoverR(rad, phi, z)*prim(IDN,k,ju+j,i);
        }
      }
    }
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
        for (int i=il; i<=iu; ++i) {
          GetCylCoord(pco,rad,phi,z,i,ju+j,k);
          prim(IDN,k,ju+j,i) = DenProfileCyl(rad,phi,z);
          vel = VelProfileCyl(rad,phi,z);
          if (pmb->porb->orbital_advection_defined)
            vel -= vK(pmb->porb, pco->x1v(i), pco->x2v(ju+j), pco->x3v(k));
          prim(IM1,k,ju+j,i) = 0.0;
          prim(IM2,k,ju+j,i) = 0.0;
          prim(IM3,k,ju+j,i) = vel;
          if (NON_BAROTROPIC_EOS)
            prim(IEN,k,ju+j,i) = PoverR(rad, phi, z)*prim(IDN,k,ju+j,i);
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values

void DiskInnerX3(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad(0.0), phi(0.0), z(0.0);
  Real vel;
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    for (int k=1; k<=ngh; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=il; i<=iu; ++i) {
          GetCylCoord(pco,rad,phi,z,i,j,kl-k);
          prim(IDN,kl-k,j,i) = DenProfileCyl(rad,phi,z);
          vel = VelProfileCyl(rad,phi,z);
          if (pmb->porb->orbital_advection_defined)
            vel -= vK(pmb->porb, pco->x1v(i), pco->x2v(j), pco->x3v(kl-k));
          prim(IM1,kl-k,j,i) = 0.0;
          prim(IM2,kl-k,j,i) = vel;
          prim(IM3,kl-k,j,i) = 0.0;
          if (NON_BAROTROPIC_EOS)
            prim(IEN,kl-k,j,i) = PoverR(rad, phi, z)*prim(IDN,kl-k,j,i);
        }
      }
    }
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    for (int k=1; k<=ngh; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=il; i<=iu; ++i) {
          GetCylCoord(pco,rad,phi,z,i,j,kl-k);
          prim(IDN,kl-k,j,i) = DenProfileCyl(rad,phi,z);
          vel = VelProfileCyl(rad,phi,z);
          if (pmb->porb->orbital_advection_defined)
            vel -= vK(pmb->porb, pco->x1v(i), pco->x2v(j), pco->x3v(kl-k));
          prim(IM1,kl-k,j,i) = 0.0;
          prim(IM2,kl-k,j,i) = 0.0;
          prim(IM3,kl-k,j,i) = vel;
          if (NON_BAROTROPIC_EOS)
            prim(IEN,kl-k,j,i) = PoverR(rad, phi, z)*prim(IDN,kl-k,j,i);
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values

void DiskOuterX3(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad(0.0), phi(0.0), z(0.0);
  Real vel;
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    for (int k=1; k<=ngh; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=il; i<=iu; ++i) {
          GetCylCoord(pco,rad,phi,z,i,j,ku+k);
          prim(IDN,ku+k,j,i) = DenProfileCyl(rad,phi,z);
          vel = VelProfileCyl(rad,phi,z);
          if (pmb->porb->orbital_advection_defined)
            vel -= vK(pmb->porb, pco->x1v(i), pco->x2v(j), pco->x3v(ku+k));
          prim(IM1,ku+k,j,i) = 0.0;
          prim(IM2,ku+k,j,i) = vel;
          prim(IM3,ku+k,j,i) = 0.0;
          if (NON_BAROTROPIC_EOS)
            prim(IEN,ku+k,j,i) = PoverR(rad, phi, z)*prim(IDN,ku+k,j,i);
        }
      }
    }
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    for (int k=1; k<=ngh; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=il; i<=iu; ++i) {
          GetCylCoord(pco,rad,phi,z,i,j,ku+k);
          prim(IDN,ku+k,j,i) = DenProfileCyl(rad,phi,z);
          vel = VelProfileCyl(rad,phi,z);
          if (pmb->porb->orbital_advection_defined)
            vel -= vK(pmb->porb, pco->x1v(i), pco->x2v(j), pco->x3v(ku+k));
          prim(IM1,ku+k,j,i) = 0.0;
          prim(IM2,ku+k,j,i) = 0.0;
          prim(IM3,ku+k,j,i) = vel;
          if (NON_BAROTROPIC_EOS)
            prim(IEN,ku+k,j,i) = PoverR(rad, phi, z)*prim(IDN,ku+k,j,i);
        }
      }
    }
  }
}

namespace
{

void DiskOpacity(MeshBlock *pmb, AthenaArray<Real> &prim)
{
  Radiation *prad = pmb->prad;
  int il = pmb->is; int jl = pmb->js; int kl = pmb->ks;
  int iu = pmb->ie; int ju = pmb->je; int ku = pmb->ke;
  il -= NGHOST;
  iu += NGHOST;
  if(ju > jl){
    jl -= NGHOST;
    ju += NGHOST;
  }
  if(ku > kl){
    kl -= NGHOST;
    ku += NGHOST;
  }

  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=il; i<=iu; ++i) {
        for (int ifr=0; ifr<prad->nfreq; ++ifr){
          Real rho  = prim(IDN,k,j,i);
          Real gast = std::max(prim(IEN,k,j,i)/rho,tfloor);
          gast = std::min(prim(IEN,k,j,i)/rho,tceiling);
          Real gastphys = gast * tunit;
          
          PolyOpacityRosseland(gastphys, kappa_a);
          PolyOpacityPlanck(gastphys, kappa_planck);
          kappa_a = kappa_a * rhounit * lunit;
          kappa_planck = kappa_planck * rhounit * lunit;

          prad->sigma_s(k,j,i,ifr) = kappa_s * rho;
          prad->sigma_a(k,j,i,ifr) = kappa_a * rho;
          prad->sigma_pe(k,j,i,ifr) = kappa_planck * rho;
          prad->sigma_p(k,j,i,ifr) =  kappa_planck * rho;
        }
      }
    }
  }
}

void Inner_rad_X1(MeshBlock *pmb, Coordinates *pco, Radiation *prad,
     const AthenaArray<Real> &w, FaceField &b, AthenaArray<Real> &ir,
      Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
  Real rad(0.0), phi(0.0), z(0.0);


  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=1; i<=ngh; ++i) {
        GetCylCoord(pco,rad,phi,z,is-i,j,k);
        Real gast = t_bkg;// PoverR(rad, phi, z);
        for(int ifr=0; ifr<prad->nfreq; ++ifr){
          for(int n=0; n<prad->nang; ++n){
            Real& miux=prad->mu(0,k,j,is,n);
            if(miux < 0.0){
              ir(k,j,is-i,ifr*prad->nang+n)
                = ir(k,j,is,ifr*prad->nang+n);
            }else{
              ir(k,j,is-i,ifr*prad->nang+n) = gast*gast*gast*gast;
            }
          }
        }
                        
      }//i
    }//j
  }//k
  return;
}


void Outer_rad_X1(MeshBlock *pmb, Coordinates *pco, Radiation *prad,
     const AthenaArray<Real> &w, FaceField &b, AthenaArray<Real> &ir,
      Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
  Real rad(0.0), phi(0.0), z(0.0);

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=1; i<=ngh; ++i) {
        GetCylCoord(pco,rad,phi,z,ie+i,j,k);
        Real gast = t_bkg; // PoverR(rad, phi, z);
          
        for(int ifr=0; ifr<prad->nfreq; ++ifr){
          for(int n=0; n<prad->nang; ++n){
            Real& miux=prad->mu(0,k,j,is,n);
            if(miux > 0.0){
              ir(k,j,ie+i,ifr*prad->nang+n)
                = ir(k,j,ie,ifr*prad->nang+n);
            }else{
              ir(k,j,ie+i,ifr*prad->nang+n) = gast*gast*gast*gast;
            }
          }
        }
      }//i
    }//j
  }//k
  return;
}


void Inner_rad_X2(MeshBlock *pmb, Coordinates *pco, Radiation *prad,
     const AthenaArray<Real> &w, FaceField &b, AthenaArray<Real> &ir,
      Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
  Real rad(0.0), phi(0.0), z(0.0);
  for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=ngh; ++j) {
      for (int i=is; i<=ie; ++i) {
        GetCylCoord(pco,rad,phi,z,i,js-j,k);
        Real gast = t_bkg; // PoverR(rad, phi, z);
          
        for(int ifr=0; ifr<prad->nfreq; ++ifr){
          for(int n=0; n<prad->nang; ++n){
            Real miuy = prad->mu(1,k,j,i,ifr*prad->nang+n);
            if(miuy < 0.0){
              ir(k,js-j,i,ifr*prad->nang+n)
                         = ir(k,js,i,ifr*prad->nang+n);
            }else{
                ir(k,js-j,i,ifr*prad->nang+n) = gast*gast*gast*gast;
            }
          }
        }
                        
      }//i
    }//j
  }//k
  return;
}

void Outer_rad_X2(MeshBlock *pmb, Coordinates *pco, Radiation *prad,
     const AthenaArray<Real> &w, FaceField &b, AthenaArray<Real> &ir,
      Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
  Real rad(0.0), phi(0.0), z(0.0);

  for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=ngh; ++j) {
      for (int i=is; i<=ie; ++i) {
        GetCylCoord(pco,rad,phi,z,i,je+j,k);
        Real gast = t_bkg; //PoverR(rad, phi, z);
        for(int ifr=0; ifr<prad->nfreq; ++ifr){
          for(int n=0; n<prad->nang; ++n){
            Real miuy = prad->mu(1,k,j,i,ifr*prad->nang+n);
            if(miuy > 0.0){
              ir(k,je+j,i,ifr*prad->nang+n)
                              = ir(k,je,i,ifr*prad->nang+n);
            }else{
                ir(k,je+j,i,ifr*prad->nang+n) = gast*gast*gast*gast;
            }
          }
        }
      }//i
    }//j
  }//k
  return;
}

void StellarHeating(MeshBlock *pmb, const Real time, const Real dt,
             const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
             const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
             AthenaArray<Real> &cons_scalar)
{
  Radiation *prad = pmb->prad;
  Coordinates *pco = pmb->pcoord;
  AthenaArray<Real> tau, x1area, vol;
  tau.NewAthenaArray(pmb->block_size.nx1+1);
  x1area.NewAthenaArray(pmb->ncells1+1);
  vol.NewAthenaArray(pmb->ncells1); // why this needs to be nx1+1 otherwise segfault... be careful for the indexing here...
  
  int lx3 = static_cast<int>(pmb->loc.lx3);
  int lx2 = static_cast<int>(pmb->loc.lx2);
  int lx1 = static_cast<int>(pmb->loc.lx1);
  int ks = pmb->ks;
  int ke = pmb->ke;
  int js = pmb->js;
  int je = pmb->je;
  int is = pmb->is;
  int ie = pmb->ie;
  for (int k = ks; k <= ke; ++k) {
    Real sinphi = std::sin(pco->x3v(k));
    for (int j = js; j <= je; ++j) {
      int tj  = lx2*pmb->block_size.nx2+(j-js);
      int tk  = lx3*pmb->block_size.nx3+(k-ks);
      pco->Face1Area(k, j, is, ie+1, x1area);
      pco->CellVolume(k, j, is, ie, vol);
      Real sintheta = std::sin(pco->x2v(j));
      Real attenuation;
      GaussianAttenuation(sintheta, sinphi, amp_shadow,
                          sigma_shadow, t_growth, time, attenuation);
      attenuation = 1;
      // std::cout << attenuation << std::endl;
      //
      Real tau_left = pmb->pmy_mesh->ruser_mesh_data[0](tk,tj,lx1);
      tau(0) = tau_left;
      for (int i = is; i<= ie; ++i){
        // step 1.
        tau(i-is+1) = pmb->ruser_meshblock_data[0](k-ks,j-js,i-is)
        + tau_left;
        // double check if x1f(i+1) or not
        Real etot = cons(IEN,k,j,i);
        Real etot0 = cons(IEN,k,j,i); // for test
        //Real total_planck = 0; // for test
        for (int ifrs=0; ifrs<len_freq; ++ifrs){
          Real f_star_right = attenuation *
                             (r_star/pco->x1f(i+1))*(r_star/pco->x1f(i+1))*planck_function(ifrs)
                             *std::exp(-tau(i-is+1)/kappa_star*stellar_opacity(ifrs));
          Real f_star_left  = attenuation *
                              (r_star/pco->x1f(i))*(r_star/pco->x1f(i))*planck_function(ifrs)
                              *std::exp(-tau(i-is)/kappa_star*stellar_opacity(ifrs));
          // step 2.
          // std::cout << "f_star_right, left " << f_star_right  << " " << f_star_left << std::endl;
          etot = etot - dt * (x1area(i+1)*f_star_right
                              - x1area(i)*f_star_left)
                            / vol(i);
          //total_planck += planck_function(ifrs);
        }
        //std::cout << total_planck << std::endl;
        
        // set a temperature ceiling
        //std::cout << j << ',' << k << ',' << i << "," << etot-etot0 << std::endl;
        Real ekin = 0.5*(SQR(cons(IM1,k,j,i))+SQR(cons(IM2,k,j,i))
                          +SQR(cons(IM3,k,j,i)))/cons(IDN,k,j,i);
        Real eint = etot-ekin;
        Real eint_ceiling = 1.0/(gamma_gas-1.0)*cons(IDN,k,j,i)*tceiling;
        Real eint_floor   = 1.0/(gamma_gas-1.0)*cons(IDN,k,j,i)*tfloor;
        eint = std::min(eint, eint_ceiling);
        eint = std::max(eint, eint_floor);
        etot = eint+ekin;
        cons(IEN,k,j,i) = etot;
      }
    }
  }
  return;
}

void PolyOpacityPlanck(const Real tgas, Real &kappa)
{
  Real log10t = std::log10(tgas);
  log10t = std::max(log10t, 1.); // floor temp: 10K
  log10t = std::min(log10t, 3.); // ceiling temp: 1000K
  kappa = -0.01241228*std::pow(log10t, 9)
       +   0.17813824*std::pow(log10t, 8)
       +  -1.08987454*std::pow(log10t, 7)
       +  3.71035294*std::pow(log10t, 6)
       + -7.5728561*std::pow(log10t, 5)
       +  9.10713297*std::pow(log10t, 4)
       + -5.99285889*std::pow(log10t, 3)
       +   2.01979676*std::pow(log10t, 2)
       +   1.52952764*std::pow(log10t, 1)
       +  -1.15919713;
  kappa = dgratio*small_grain_ratio*std::pow(10, kappa);
}


void PolyOpacityRosseland(const Real tgas, Real &kappa)
{
  Real log10t = std::log10(tgas);
  log10t = std::max(log10t, 1.); // floor temp: 10K
  log10t = std::min(log10t, 3.); // ceiling temp: 1000K
  kappa = -0.07075475*std::pow(log10t, 9)
       +   0.93913158*std::pow(log10t, 8)
       +  -5.13129509*std::pow(log10t, 7)
       +  14.96480641*std::pow(log10t, 6)
       + -25.29213858*std::pow(log10t, 5)
       +  25.09673957*std::pow(log10t, 4)
       + -14.02904307*std::pow(log10t, 3)
       +   4.01559596*std::pow(log10t, 2)
       +   1.31106774*std::pow(log10t, 1)
       +  -1.4164232;
  kappa = dgratio*small_grain_ratio*std::pow(10, kappa);
}

void GaussianAttenuation(const Real sintheta, const Real sinphi,
                         const Real amp, const Real sigma, const Real t_growth, Real time, Real &attenuation)
{
  Real y = sintheta * sinphi;
  Real theta_y = std::acos(y); 
  attenuation = 1 - amp * std::exp(-(theta_y-PI/2.)*(theta_y-PI/2.)*(theta_y-PI/2.)*(theta_y-PI/2.) / (sigma*sigma));
  if (time < t_growth){
    Real sint = std::sin(PI/2. * time/t_growth);
    attenuation = sint*sint*attenuation;
  }
}

}//namespace


