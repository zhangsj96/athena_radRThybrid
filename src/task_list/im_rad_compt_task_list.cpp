//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file fft_grav_task_list.cpp
//! \brief function implementation for FFTGravitySolverTaskList

// C headers

// C++ headers
#include <iostream>   // endl
#include <sstream>    // sstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"
#include "../mesh/mesh.hpp"
#include "../radiation/radiation.hpp"
#include "../radiation/implicit/radiation_implicit.hpp"
#include "../radiation/integrators/rad_integrators.hpp"
#include "../hydro/hydro.hpp"
#include "im_rad_task_list.hpp"

//----------------------------------------------------------------------------------------
//! IMRadTaskList constructor

IMRadComptTaskList::IMRadComptTaskList(Mesh *pm) {
  pmy_mesh = pm;
  // Now assemble list of tasks for each stage of time integrator
  {using namespace IMRadComptTaskNames; // NOLINT (build/namespace)
    // compute hydro fluxes, integrate hydro variables
    AddTask(CAL_COMPT,NONE);  
    AddTask(SEND_RAD_BND,CAL_COMPT);
    AddTask(RECV_RAD_BND,CAL_COMPT);
    AddTask(SETB_RAD_BND,(RECV_RAD_BND|SEND_RAD_BND));
    if(pm->shear_periodic){
      AddTask(SEND_RAD_SH,SETB_RAD_BND);
      AddTask(RECV_RAD_SH,SEND_RAD_SH|RECV_RAD_BND);
    }
    TaskID setb = SETB_RAD_BND;
    if(pm->shear_periodic)
      setb=(setb|RECV_RAD_SH);
    if(pm->multilevel){
      AddTask(PRLN_RAD_BND,setb);
      AddTask(RAD_PHYS_BND,PRLN_RAD_BND);
    }else{
      AddTask(RAD_PHYS_BND,setb);
    }
    AddTask(CLEAR_RAD, RAD_PHYS_BND);
  } // end of using namespace block

}

//----------------------------------------------------------------------------------------
//! \fn void FFTGravitySolverTaskList::AddTask(const TaskID& id, const TaskID& dep)
//! \brief Sets id and dependency for "ntask" member of task_list_ array, then iterates
//! value of ntask.

void IMRadComptTaskList::AddTask(const TaskID& id, const TaskID& dep) {
  task_list_[ntasks].task_id=id;
  task_list_[ntasks].dependency=dep;

  using namespace IMRadComptTaskNames; // NOLINT (build/namespace)
  if (id == CLEAR_RAD) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (IMRadTaskList::*)(MeshBlock*)>
        (&IMRadComptTaskList::ClearRadBoundary);
  } else if (id == SEND_RAD_BND) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (IMRadTaskList::*)(MeshBlock*)>
        (&IMRadComptTaskList::SendRadBoundary);
  } else if (id == RECV_RAD_BND) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (IMRadTaskList::*)(MeshBlock*)>
        (&IMRadComptTaskList::ReceiveRadBoundary);
  } else if (id == SETB_RAD_BND) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (IMRadTaskList::*)(MeshBlock*)>
        (&IMRadComptTaskList::SetRadBoundary);
  } else if (id == RAD_PHYS_BND) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (IMRadTaskList::*)(MeshBlock*)>
        (&IMRadComptTaskList::PhysicalBoundary);
   }else if (id == PRLN_RAD_BND) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (IMRadTaskList::*)(MeshBlock*)>
        (&IMRadComptTaskList::ProlongateBoundary);
  } else if (id == SEND_RAD_SH) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (IMRadTaskList::*)(MeshBlock*)>
        (&IMRadComptTaskList::SendRadBoundaryShear);
  } else if (id == RECV_RAD_SH) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (IMRadTaskList::*)(MeshBlock*)>
        (&IMRadComptTaskList::ReceiveRadBoundaryShear);
  } else if (id == CAL_COMPT) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (IMRadTaskList::*)(MeshBlock*)>
        (&IMRadComptTaskList::CalComptTerms);
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in IMRadTaskList::AddTask" << std::endl
        << "Invalid Task is specified" << std::endl;
    ATHENA_ERROR(msg);
  }
  ntasks++;
  return;
}


TaskStatus IMRadComptTaskList::CalComptTerms(MeshBlock *pmb) {

  Radiation *prad = pmb->prad;
  Hydro *ph = pmb->phydro;
  
  prad->pradintegrator->AddMultiGroupCompt(pmb, dt, ph->u, prad->ir);


  return TaskStatus::success;
  
}



TaskStatus IMRadComptTaskList::ClearRadBoundary(MeshBlock *pmb) {
  pmb->prad->rad_bvar.ClearBoundary(BoundaryCommSubset::radiation);
  return TaskStatus::success;
}

TaskStatus IMRadComptTaskList::SendRadBoundary(MeshBlock *pmb) {
  pmb->prad->rad_bvar.SendBoundaryBuffers();
  return TaskStatus::success;
}

TaskStatus IMRadComptTaskList::SendRadBoundaryShear(MeshBlock *pmb) {
  pmb->prad->rad_bvar.SendShearingBoxBoundaryBuffers();
  return TaskStatus::success;
}


TaskStatus IMRadComptTaskList::ReceiveRadBoundary(MeshBlock *pmb) {
  bool ret = pmb->prad->rad_bvar.ReceiveBoundaryBuffers();
  if (!ret)
    return TaskStatus::fail;
  return TaskStatus::success;
}

TaskStatus IMRadComptTaskList::ReceiveRadBoundaryShear(MeshBlock *pmb) {
  bool ret = pmb->prad->rad_bvar.ReceiveShearingBoxBoundaryBuffers();
  if(ret){
    pmb->prad->rad_bvar.SetShearingBoxBoundaryBuffers();
    return TaskStatus::success;
  }else{
    return TaskStatus::fail;
  }
}


TaskStatus IMRadComptTaskList::SetRadBoundary(MeshBlock *pmb) {
  pmb->prad->rad_bvar.SetBoundaries();
  return TaskStatus::success;
}



void IMRadComptTaskList::StartupTaskList(MeshBlock *pmb) {
  pmb->prad->rad_bvar.StartReceiving(BoundaryCommSubset::radiation);
  if(pmy_mesh->shear_periodic)
    pmb->prad->rad_bvar.StartReceivingShear(BoundaryCommSubset::radiation);
  return;
}


