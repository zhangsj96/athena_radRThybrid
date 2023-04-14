//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file grav_task_list.cpp
//! \brief function implementation for GravityBoundaryTaskList

// C headers

// C++ headers
#include <iostream>   // endl
#include <sstream>    // sstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../gravity/gravity.hpp"
#include "../hydro/hydro.hpp"
#include "../hydro/srcterms/hydro_srcterms.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "grav_task_list.hpp"
#include "task_list.hpp"

//----------------------------------------------------------------------------------------
//! GravityBoundaryTaskList constructor
GravityBoundaryTaskList::GravityBoundaryTaskList(ParameterInput *pin, Mesh *pm) {
  integrator = pin->GetOrAddString("time", "integrator", "vl2");

  // Read a flag for orbital advection
  ORBITAL_ADVECTION = (pm->orbital_advection != 0)? true : false;

  // Read a flag for shear periodic
  SHEAR_PERIODIC = pm->shear_periodic;

  if (SHEAR_PERIODIC) {
    if (integrator == "vl2") {
      //! \note `integrator == "vl2"`
      //! - VL: second-order van Leer integrator (Stone & Gardiner, NewA 14, 139 2009)
      //! - Simple predictor-corrector scheme similar to MUSCL-Hancock
      //! - Expressed in 2S or 3S* algorithm form

      // set number of stages and time coeff.
      if (ORBITAL_ADVECTION) {
        std::stringstream msg;
        msg << "### FATAL ERROR in GravityBoundaryTaskList constructor" << std::endl
            << "Gravity is not tested with orbital advection" << std::endl;
        ATHENA_ERROR(msg);
      } else { // w/o orbital advection
        nstages = 2;
        // To be fully consistent with the TimeIntegratorTaskList, sbeta and ebeta
        // must be set identical to the corresponding values in
        // TimeIntegratorTaskList::stage_wghts.
        sbeta[0] = 0.0;
        ebeta[0] = 0.5;
        sbeta[1] = 0.5;
        ebeta[1] = 1.0;
      }
    } else {
      std::stringstream msg;
      msg << "### FATAL ERROR in GravityBoundaryTaskList constructor" << std::endl
          << "Gravity is not tested with integrator=" << integrator
          << " in shearing-periodic BC" << std::endl;
      ATHENA_ERROR(msg);
    }
  }

  // Now assemble list of tasks for each stage of time integrator
  {using namespace GravityBoundaryTaskNames; // NOLINT (build/namespace)
    // compute hydro fluxes, integrate hydro variables
    AddTask(SEND_GRAV_BND,NONE);
    AddTask(RECV_GRAV_BND,NONE);
    AddTask(SETB_GRAV_BND,(RECV_GRAV_BND|SEND_GRAV_BND));
    if (pm->multilevel) {
      AddTask(PROLONG_GRAV_BND,SETB_GRAV_BND);
      AddTask(GRAV_PHYS_BND,PROLONG_GRAV_BND);
    } else if (SHEAR_PERIODIC) { // Shearingbox BC for Gravity
      AddTask(SEND_GRAV_SH,SETB_GRAV_BND);
      AddTask(RECV_GRAV_SH,SETB_GRAV_BND);
      AddTask(GRAV_PHYS_BND,(SEND_GRAV_SH|RECV_GRAV_SH));
    } else {
      AddTask(GRAV_PHYS_BND,SETB_GRAV_BND);
    }
    AddTask(CLEAR_GRAV, GRAV_PHYS_BND);
  } // end of using namespace block
}

//----------------------------------------------------------------------------------------
//! \fn void GravityBoundaryTaskList::AddTask(const TaskID& id, const TaskID& dep)
//! \brief Sets id and dependency for "ntask" member of task_list_ array, then iterates
//! value of ntask.

void GravityBoundaryTaskList::AddTask(const TaskID& id, const TaskID& dep) {
  task_list_[ntasks].task_id=id;
  task_list_[ntasks].dependency=dep;
  task_list_[ntasks].task_name.assign("");

  using namespace GravityBoundaryTaskNames; // NOLINT (build/namespace)
  if (id == CLEAR_GRAV) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&GravityBoundaryTaskList::ClearGravityBoundary);
    task_list_[ntasks].lb_time = false;
    task_list_[ntasks].task_name.append("ClearGravityBoundary");
  } else if (id == SEND_GRAV_BND) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&GravityBoundaryTaskList::SendGravityBoundary);
    task_list_[ntasks].lb_time = true;
    task_list_[ntasks].task_name.append("SendGravityBoundary");
  } else if (id == RECV_GRAV_BND) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&GravityBoundaryTaskList::ReceiveGravityBoundary);
    task_list_[ntasks].lb_time = false;
    task_list_[ntasks].task_name.append("ReceiveGravityBoundary");
  } else if (id == SETB_GRAV_BND) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&GravityBoundaryTaskList::SetGravityBoundary);
  } else if (id == PROLONG_GRAV_BND) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&GravityBoundaryTaskList::ProlongateGravityBoundary);
  } else if (id == GRAV_PHYS_BND) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&GravityBoundaryTaskList::PhysicalBoundary);
    task_list_[ntasks].lb_time = true;
    task_list_[ntasks].task_name.append("PhysicalBoundary");
  } else if (id == SEND_GRAV_SH) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&GravityBoundaryTaskList::SendGravityShear);
  } else if (id == RECV_GRAV_SH) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&GravityBoundaryTaskList::ReceiveGravityShear);
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in GravityBoundaryTaskList::AddTask" << std::endl
        << "Invalid Task is specified" << std::endl;
    ATHENA_ERROR(msg);
  }
  ntasks++;
  return;
}

void GravityBoundaryTaskList::StartupTaskList(MeshBlock *pmb, int stage) {
  // Mimics TimeIntegratorTaskList::StartupTaskList
  if (SHEAR_PERIODIC) {
    Real dt_fc   = pmb->pmy_mesh->dt*sbeta[stage-1];
    Real dt_int  = pmb->pmy_mesh->dt*ebeta[stage-1];
    Real time = pmb->pmy_mesh->time;
    pmb->pbval->ComputeShear(time+dt_fc, time+dt_int);
  }

  pmb->pgrav->gbvar.StartReceiving(BoundaryCommSubset::all);
  if (SHEAR_PERIODIC) {
    pmb->pgrav->gbvar.StartReceivingShear(BoundaryCommSubset::all);
  }
  return;
}

TaskStatus GravityBoundaryTaskList::ClearGravityBoundary(MeshBlock *pmb, int stage) {
  pmb->pgrav->gbvar.ClearBoundary(BoundaryCommSubset::all);
  return TaskStatus::success;
}

TaskStatus GravityBoundaryTaskList::SendGravityBoundary(MeshBlock *pmb, int stage) {
  if (pmb->pgrav->fill_ghost)
    pmb->pgrav->SaveFaceBoundaries();
  pmb->pgrav->gbvar.SendBoundaryBuffers();
  return TaskStatus::success;
}

TaskStatus GravityBoundaryTaskList::ReceiveGravityBoundary(MeshBlock *pmb,
                                                               int stage) {
  bool ret = pmb->pgrav->gbvar.ReceiveBoundaryBuffers();
  if (!ret)
    return TaskStatus::fail;
  return TaskStatus::success;
}

TaskStatus GravityBoundaryTaskList::SendGravityShear(MeshBlock *pmb, int stage) {
  if (stage <= nstages) {
    pmb->pgrav->gbvar.SendShearingBoxBoundaryBuffers();
  } else {
    return TaskStatus::fail;
  }
  return TaskStatus::success;
}

TaskStatus GravityBoundaryTaskList::ReceiveGravityShear(MeshBlock *pmb, int stage) {
  bool ret;
  ret = false;
  if (stage <= nstages) {
    ret = pmb->pgrav->gbvar.ReceiveShearingBoxBoundaryBuffers();
  } else {
    return TaskStatus::fail;
  }
  if (ret) {
    pmb->pgrav->gbvar.SetShearingBoxBoundaryBuffers();
    return TaskStatus::success;
  } else {
    return TaskStatus::fail;
  }
}

TaskStatus GravityBoundaryTaskList::SetGravityBoundary(MeshBlock *pmb, int stage) {
  pmb->pgrav->gbvar.SetBoundaries();
  return TaskStatus::success;
}

TaskStatus GravityBoundaryTaskList::ProlongateGravityBoundary(MeshBlock *pmb,
                                                              int stage) {
  pmb->pbval->ProlongateGravityBoundaries(pmb->pmy_mesh->time, 0.0);
  return TaskStatus::success;
}

TaskStatus GravityBoundaryTaskList::PhysicalBoundary(MeshBlock *pmb, int stage) {
  if (pmb->pgrav->fill_ghost) {
    pmb->pgrav->RestoreFaceBoundaries();
    pmb->pgrav->ExpandPhysicalBoundaries();
  }
  return TaskStatus::next;
}
