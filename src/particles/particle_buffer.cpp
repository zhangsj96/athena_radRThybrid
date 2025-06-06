//======================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
//======================================================================================
//! \file particle_buffer.cpp
//! \brief implements ParticleBuffer class for communication of particles.
//======================================================================================

// C++ standard libraries
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>

// Athena++ headers
#include "particle_buffer.hpp"

//--------------------------------------------------------------------------------------
//! \fn ParticleBuffer::ParticleBuffer()
//! \brief initiates a default instance of ParticleBuffer.

ParticleBuffer::ParticleBuffer() {
  ibuf = NULL;
  rbuf = NULL;
  nparmax_ = npar_ = 0;
#ifdef MPI_PARALLEL
  reqn = reqi = reqr = MPI_REQUEST_NULL;
  flagn = flagi = flagr = 0;
  tag = -1;
#endif
}

//--------------------------------------------------------------------------------------
//! \fn ParticleBuffer::ParticleBuffer(int nparmax0, int nint, int nreal)
//! \brief initiates a new instance of ParticleBuffer with nparmax = nparmax0.

ParticleBuffer::ParticleBuffer(int nparmax0, int nint, int nreal) {
  // Sanity check
  if (nparmax0 <= 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [ParticleBuffer::ParticleBuffer]" << std::endl
        << "Invalid nparmax0 = " << nparmax0 << std::endl;
    ATHENA_ERROR(msg);

    ibuf = NULL;
    rbuf = NULL;
    nparmax_ = npar_ = 0;
    return;
  }

  // Initialize the instance variables.
  nparmax_ = nparmax0;
  ibuf = new int[nint * nparmax_];
  rbuf = new Real[nreal * nparmax_];
  npar_ = 0;
#ifdef MPI_PARALLEL
  reqn = reqi = reqr = MPI_REQUEST_NULL;
  flagn = flagi = flagr = 0;
  tag = -1;
#endif
}

//--------------------------------------------------------------------------------------
//! \fn ParticleBuffer::ParticleBuffer()
//! \brief destroys an instance of ParticleBuffer.

ParticleBuffer::~ParticleBuffer() {
  if (ibuf != NULL) delete [] ibuf;
  if (rbuf != NULL) delete [] rbuf;
#ifdef MPI_PARALLEL
  if (reqn != MPI_REQUEST_NULL) MPI_Request_free(&reqn);
  if (reqi != MPI_REQUEST_NULL) MPI_Request_free(&reqi);
  if (reqr != MPI_REQUEST_NULL) MPI_Request_free(&reqr);
#endif
}

//--------------------------------------------------------------------------------------
//! \fn void ParticleBuffer::Reallocate(int new_nparmax, int nint, int nreal)
//! \brief reallocates the buffers; the old content is preserved.

void ParticleBuffer::Reallocate(int new_nparmax, int nint, int nreal) {
  // Sanity check
  if (new_nparmax <= 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [ParticleBuffer::Reallocate]" << std::endl
        << "Invalid new_nparmax = " << new_nparmax << std::endl;
    ATHENA_ERROR(msg);
    return;
  }
  if (new_nparmax < npar_) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [ParticleBuffer::Reallocate]" << std::endl
        << "new_nparmax = " << new_nparmax << " < npar = " << npar_ << std::endl;
    ATHENA_ERROR(msg);
    return;
  }
#ifdef MPI_PARALLEL
  if (reqn != MPI_REQUEST_NULL || reqi != MPI_REQUEST_NULL || reqr != MPI_REQUEST_NULL) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [ParticleBuffer::Reallocate]" << std::endl
        << "MPI requests are active. " << std::endl;
    ATHENA_ERROR(msg);
    return;
  }
#endif

  // Allocate new space.
  int *ibuf_new = new int[nint * new_nparmax];
  Real *rbuf_new = new Real[nreal * new_nparmax];

  // Move existing data
  if ((npar_ > 0)&&(nparmax_ > 0)) {
    std::memcpy(ibuf_new, ibuf, nint * npar_ * sizeof(int));
    std::memcpy(rbuf_new, rbuf, nreal * npar_ * sizeof(Real));
  }
  nparmax_ = new_nparmax;

  // Delete old space.
  if (ibuf != NULL) delete [] ibuf;
  if (rbuf != NULL) delete [] rbuf;
  ibuf = ibuf_new;
  rbuf = rbuf_new;
}
