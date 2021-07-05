import numpy as np

# The stuff commented out here was used in the older c++ code that I ported this from.
# See further down for the actual python code.

#///////////////////////////////////////////////////////////////////////////////////////////////////
#/// \file formic/lbfgs.h
#///
#/// \brief   header file for the LBFGS method
#///
#///////////////////////////////////////////////////////////////////////////////////////////////////
#
##ifndef FORMIC_LBFGS_HEADER
##define FORMIC_LBFGS_HEADER
#
##include<vector>
##include<numeric>
##include<cassert>
##include<algorithm>
##include<cmath>
#
##include<boost/math/special_functions/fpclassify.hpp>
#
##include<formic/mpi/interface.h>
##include<formic/output/output.h>
##include<formic/exception.h>
##include<formic/lapack/interface.h>
##include<formic/numeric/numeric.h>
#
#namespace formic {
#
#//  template <class Manager>
#//  void lbfgs(const int n,
#//             double & f,
#//             double * const x,
#//             double * const g,
#//             const int max_iter,
#//             const int max_hist,
#//             Manager & man) {
#//
#//    std::vector<double> rho(max_hist, 0.0);
#//    std::vector<double> alpha(max_hist, 0.0);
#//    std::vector<double> beta(max_hist, 0.0);
#//
#//    std::vector<double> old_x_vec(n, 0.0);
#//    std::vector<double> old_g_vec(n, 0.0);
#//    double * const old_x = &old_x_vec.at(0);
#//    double * const old_g = &old_g_vec.at(0);
#//
#//    for (int iter = 0; iter < max_iter; iter++) {
#//
#//      // save old parameter and gradient vectors
#//      if ( iter > 0 ) {
#//        formic::xcopy(n, x, 1, old_x, 1);
#//        formic::xcopy(n, g, 1, old_g, 1);
#//      }
#//
#//      // get the value
#//      man.get_val(n, x, f);
#//
#//      // get the gradient
#//      man.get_grd(n, x, g);
#//
#//      // compute differences
#//      if ( iter > 0 ) {
#//        formic::xscal(n, -1.0, old_x, 1);
#//        formic::xscal(n, -1.0, old_g, 1);
#//      }
#//
#//      // compute the rho vector
#//      for (int i = 1; i < pmtr_hist.size() && i <= userinp.get<int>("bfgs_hist_len"); i++) {
#//        //rho.at(i) += formic::xdotc(pmtr_hist.at(i-1)->size(), &pmtr_hist.at(i-1)->at(0), 1, &grad_hist.at(i-1)->at(0), 1);
#//        //rho.at(i) -= formic::xdotc(pmtr_hist.at(i-1)->size(), &pmtr_hist.at(i-0)->at(0), 1, &grad_hist.at(i-1)->at(0), 1);
#//        //rho.at(i) -= formic::xdotc(pmtr_hist.at(i-1)->size(), &pmtr_hist.at(i-1)->at(0), 1, &grad_hist.at(i-0)->at(0), 1);
#//        //rho.at(i) += formic::xdotc(pmtr_hist.at(i-1)->size(), &pmtr_hist.at(i-0)->at(0), 1, &grad_hist.at(i-0)->at(0), 1);
#//        //rho.at(i) = 1.0 / rho.at(i);
#//        formic::xcopy(pdiff.size(),       &pmtr_hist.at(i-1)->at(0), 1, &pdiff.at(0), 1);
#//        formic::xaxpy(pdiff.size(), -1.0, &pmtr_hist.at(i-0)->at(0), 1, &pdiff.at(0), 1);
#//        formic::xcopy(gdiff.size(),       &grad_hist.at(i-1)->at(0), 1, &gdiff.at(0), 1);
#//        formic::xaxpy(gdiff.size(), -1.0, &grad_hist.at(i-0)->at(0), 1, &gdiff.at(0), 1);
#//        rho.at(i) = 1.0 / formic::xdotc(pdiff.size(), &pdiff.at(0), 1, &gdiff.at(0), 1);
#//        if (formic::mpi::rank() == 0)
#//          formic::of << boost::format("rho[%2i] = %.2e") % i % rho.at(i) << std::endl;
#//      }
#//
#//      // prepare space for the alpha and beta vectors
#//
#//      // compute the update
#//      std::vector<double> update(**grad_hist.begin());
#//      for (int i = 1; i < pmtr_hist.size() && i <= userinp.get<int>("bfgs_hist_len"); i++) {
#//        //alpha.at(i) += rho.at(i) * formic::xdotc(update.size(), &pmtr_hist.at(i-1)->at(0), 1, &update.at(0), 1);
#//        //alpha.at(i) -= rho.at(i) * formic::xdotc(update.size(), &pmtr_hist.at(i-0)->at(0), 1, &update.at(0), 1);
#//        //formic::xaxpy(update.size(), -alpha.at(i), &grad_hist.at(i-1)->at(0), 1, &update.at(0), 1);
#//        //formic::xaxpy(update.size(),  alpha.at(i), &grad_hist.at(i-0)->at(0), 1, &update.at(0), 1);
#//        formic::xcopy(pdiff.size(),       &pmtr_hist.at(i-1)->at(0), 1, &pdiff.at(0), 1);
#//        formic::xaxpy(pdiff.size(), -1.0, &pmtr_hist.at(i-0)->at(0), 1, &pdiff.at(0), 1);
#//        formic::xcopy(gdiff.size(),       &grad_hist.at(i-1)->at(0), 1, &gdiff.at(0), 1);
#//        formic::xaxpy(gdiff.size(), -1.0, &grad_hist.at(i-0)->at(0), 1, &gdiff.at(0), 1);
#//        alpha.at(i) = rho.at(i) * formic::xdotc(update.size(), &pdiff.at(0), 1, &update.at(0), 1);
#//        formic::xaxpy(update.size(), -alpha.at(i), &gdiff.at(0), 1, &update.at(0), 1);
#//      }
#//      for (int i = std::min(int(pmtr_hist.size() - 1), userinp.get<int>("bfgs_hist_len")); i > 0; i--) {
#//        //beta.at(i) += rho.at(i) * formic::xdotc(update.size(), &grad_hist.at(i-1)->at(0), 1, &update.at(0), 1);
#//        //beta.at(i) -= rho.at(i) * formic::xdotc(update.size(), &grad_hist.at(i-0)->at(0), 1, &update.at(0), 1);
#//        //formic::xaxpy(update.size(), alpha.at(i) - beta.at(i), &pmtr_hist.at(i-1)->at(0), 1, &update.at(0), 1);
#//        //formic::xaxpy(update.size(), beta.at(i) - alpha.at(i), &pmtr_hist.at(i-0)->at(0), 1, &update.at(0), 1);
#//        formic::xcopy(pdiff.size(),       &pmtr_hist.at(i-1)->at(0), 1, &pdiff.at(0), 1);
#//        formic::xaxpy(pdiff.size(), -1.0, &pmtr_hist.at(i-0)->at(0), 1, &pdiff.at(0), 1);
#//        formic::xcopy(gdiff.size(),       &grad_hist.at(i-1)->at(0), 1, &gdiff.at(0), 1);
#//        formic::xaxpy(gdiff.size(), -1.0, &grad_hist.at(i-0)->at(0), 1, &gdiff.at(0), 1);
#//        beta.at(i) = rho.at(i) * formic::xdotc(update.size(), &gdiff.at(0), 1, &update.at(0), 1);
#//        formic::xaxpy(update.size(), alpha.at(i)-beta.at(i), &pdiff.at(0), 1, &update.at(0), 1);
#//      }
#//      formic::xscal(update.size(), -1.0, &update.at(0), 1);
#//
#//      // convert the update to complex form if necessary
#//      std::vector<S> cplx_update(co->n_der(), formic::zero(S()));
#//      local_funcs::real_to_cplx(co->n_der(), &update.at(0), &cplx_update.at(0));
#//
#//      // perform a line search along the update direction and remember the optimum step size
#//      formic::mpi::bcast(_step_size);
#//      double step_arg = _step_size;
#//      formic::fqmc::line_search(userinp, burn_len, samp_len, start_config, start_rng, &cplx_update.at(0), userinp.get<double>("update_max_sdev"), old_co, co, step_arg);
#//      if (step_arg > 0.0)
#//        _step_size = step_arg;
#//
#//    }
#//
#//  }
#
#  template<class W>
#  inline void quadratic_line_search(W & worker,
#                                    const int n,
#                                    const double f0,
#                                    const double * const x0,
#                                    const double * const p,
#                                    double * const x1,
#                                    double & step_arg,
#                                    const bool bcast_step_size = true) {
#
#    //// start a timer
#    //formic::start_timer("lbgfs loose line search");
#
#    // get the initial value
#    const double init_energy = f0;
#
#    // print initial value
#    if (formic::mpi::rank() == 0) {
#      formic::of << boost::format("for a step size of %16.8e") % 0.0;
#      formic::of << boost::format("      val = %20.12f") % init_energy << std::endl << std::endl;
#    }
#
#    // initialize line search variables
#    std::vector<double> step_sizes(1, 0.0);
#    std::vector<double> step_energies(1, init_energy);
#    double best_step_size = 0.0;
#    double best_step_energy = init_energy;
#    int best_step = 0;
#    int step_i = 0;
#    int n_refine = 0;
#
#    for (int snum = 1; snum <= 20; snum++) {
#
#      double new_step_size = -1.0;
#      bool to_break = false;
#
#      // choose what to do next
#      if ( step_sizes.size() < 2 ) {
#        new_step_size = step_arg;
#        if (formic::mpi::rank() == 0)
#          formic::of << boost::format("trying first step...") << std::endl << std::endl;
#      } else if ( step_sizes.size() < 3 ) {
#        new_step_size = step_sizes.at(step_i) * ( best_step_energy == init_energy ? 0.45 : 1.3 );
#        if (formic::mpi::rank() == 0)
#          formic::of << boost::format("trying second step...") << std::endl << std::endl;
#      } else if ( best_step == 0 ) {
#        new_step_size = step_sizes.at(1) * 0.5;
#        if (formic::mpi::rank() == 0)
#          formic::of << boost::format("so far all steps are too long, so we will try a shorter step") << std::endl << std::endl;
#      } else if ( best_step == step_sizes.size()-1
#                  && *step_energies.rbegin() < *(step_energies.rbegin()+1)
#                     + ( *step_sizes.rbegin() - *(step_sizes.rbegin()+1) ) * ( *(step_energies.rbegin()+1) - *(step_energies.rbegin()+2) ) / ( *(step_sizes.rbegin()+1) - *(step_sizes.rbegin()+2) )
#                ) {
#        // ez < ey + ( z - y ) * ( ey - ex ) / ( y - x )
#        if (formic::mpi::rank() == 0)
#          formic::of << boost::format("stopping line search due to negative curvature") << std::endl << std::endl;
#        to_break = true;
#      } else {
#        if ( best_step == step_sizes.size()-1 )
#          new_step_size = formic::porabola_min_max(&step_sizes.at(best_step-2), &step_energies.at(best_step-2));
#        else
#          new_step_size = formic::porabola_min_max(&step_sizes.at(best_step-1), &step_energies.at(best_step-1));
#        new_step_size = std::min(new_step_size, 2.0 * step_sizes.at(best_step));
#        if (formic::mpi::rank() == 0)
#          formic::of << boost::format("choose final step based on a quadratic fit") << std::endl << std::endl;
#      }
#
#      formic::mpi::bcast(to_break);
#      if (to_break)
#        break;
#
#      // if requested, broadcast the step size to all processes
#      if (bcast_step_size)
#        formic::mpi::bcast(new_step_size);
#
#      // prepare the next step
#      for (step_i = 0; step_i < step_sizes.size(); step_i++) {
#        if ( new_step_size < step_sizes.at(step_i) )
#          break;
#      }
#      step_sizes.insert(step_sizes.begin()+step_i, new_step_size);
#      step_energies.insert(step_energies.begin()+step_i, 1.0e100);
#      if ( step_i <= best_step )
#        best_step++;
#
#      // compute new value
#      if (formic::mpi::rank() == 0) {
#        formic::xcopy(n, x0, 1, x1, 1);
#        formic::xaxpy(n, step_sizes.at(step_i), p, 1, x1, 1);
#      }
#      worker.get_value(n, x1, step_energies.at(step_i));
#      if (formic::mpi::rank() == 0) {
#        formic::of << boost::format("for a step size of %16.8e") % step_sizes.at(step_i);
#        formic::of << boost::format("      val = %20.12f") % step_energies.at(step_i) << std::endl << std::endl;
#      }
#
#      // if this was the best step size so far, save it
#      if ( step_energies.at(step_i) < best_step_energy ) {
#        best_step = step_i;
#        best_step_size = step_sizes.at(step_i);
#        best_step_energy = step_energies.at(step_i);
#      }
#
#      formic::mpi::bcast(step_sizes);
#      formic::mpi::bcast(step_energies);
#      formic::mpi::bcast(best_step);
#      formic::mpi::bcast(best_step_size);
#      formic::mpi::bcast(best_step_energy);
#
#      // check if we are finished
#      if ( step_sizes.size() > 3 && best_step != 0 )
#        break;
#
#    }
#
#    if (formic::mpi::rank() == 0) {
#
#      // if a zero step was best, do not apply any update
#      if ( best_step == 0 ) {
#
#        formic::xcopy(n, x0, 1, x1, 1);
#
#        formic::of << "The line search failed to find a step that lowers the energy.  No update was applied." << std::endl << std::endl;
#
#      // otherwise, apply the update corresponding to the best step size
#      } else {
#
#        formic::xcopy(n, x0, 1, x1, 1);
#        formic::xaxpy(n, best_step_size, p, 1, x1, 1);
#
#        if (formic::mpi::rank() == 0)
#          formic::of << boost::format("The line search's best step size was %12.6f with a value of %20.12f") % best_step_size % best_step_energy
#                     << std::endl
#                     << boost::format("The corresponding update has been applied.")
#                     << std::endl
#                     << std::endl;
#
#      }
#
#    }
#
#    //// stop the timer
#    //formic::stop_timer("lbgfs loose line search");
#
#    // return the step size used
#    step_arg = best_step_size;
#
#  }
#
#  template<class W>
#  inline void lbfgs_loose_line_search(W & worker,
#                                      const int n,
#                                      const double * const x0,
#                                      const double * const p,
#                                      double * const x1,
#                                      double & step_arg,
#                                      const bool bcast_step_size = true) {
#
#    //// start a timer
#    //formic::start_timer("lbgfs loose line search");
#
#    // get the initial value
#    double init_energy;
#    worker.get_value(n, x0, init_energy);
#
#    // print initial value
#    if (formic::mpi::rank() == 0) {
#      formic::of << boost::format("for a step size of %16.8e") % 0.0;
#      formic::of << boost::format("      val = %20.12f") % init_energy << std::endl << std::endl;
#    }
#
#    // initialize line search variables
#    std::vector<double> step_sizes(1, 0.0);
#    std::vector<double> step_energies(1, init_energy);
#    double best_step_size = 0.0;
#    double best_step_energy = init_energy;
#    int best_step = 0;
#    int step_i = 0;
#    int n_refine = 0;
#
#    for (int snum = 0; snum < 30; snum++) {
#
#      double new_step_size = -1.0;
#
#      // choose what to do next
#      if ( step_sizes.size() < 2 ) {
#        new_step_size = step_arg;
#        if (formic::mpi::rank() == 0)
#          formic::of << boost::format("trying first step...") << std::endl << std::endl;
#      } else if ( step_sizes.size() < 3 ) {
#        new_step_size = step_sizes.at(step_i) * ( best_step_energy == init_energy ? 0.5 : 1.3 );
#        if (formic::mpi::rank() == 0)
#          formic::of << boost::format("trying second step...") << std::endl << std::endl;
#      } else if ( best_step == 0 ) {
#        new_step_size = step_sizes.at(1) * 0.5;
#        if (formic::mpi::rank() == 0)
#          formic::of << boost::format("so far all steps are too long, so we will try a shorter step") << std::endl << std::endl;
#      } else if ( best_step == step_sizes.size()-1 ) {
#        new_step_size = best_step_size * 1.3;
#        if (formic::mpi::rank() == 0)
#          formic::of << boost::format("best step so far was longest step, so we will try an even longer step") << std::endl << std::endl;
#      } else {
#        n_refine++;
#        new_step_size = formic::porabola_min_max(&step_sizes.at(best_step-1), &step_energies.at(best_step-1));
#        // make sure the new step size is well behaved
#        if ( new_step_size < 0.0 || boost::math::isnan(new_step_size) ) {
#          break; // stop if we have found a negative number or a nan
#        } else if ( boost::math::isinf(new_step_size) || new_step_size > 1.4 * (*step_sizes.rbegin()) ) {
#          new_step_size = 1.4 * (*step_sizes.rbegin());
#        } else if ( new_step_size < 0.6 * step_sizes.at(1) ) {
#          new_step_size = 0.6 * step_sizes.at(1);
#        }
#        if (formic::mpi::rank() == 0)
#          formic::of << boost::format("refining the local minima found during the line search") << std::endl << std::endl;
#      }
#
#      // if requested, broadcast the step size to all processes
#      if (bcast_step_size)
#        formic::mpi::bcast(new_step_size);
#
#      // prepare the next step
#      for (step_i = 0; step_i < step_sizes.size(); step_i++) {
#        if ( new_step_size < step_sizes.at(step_i) )
#          break;
#      }
#      step_sizes.insert(step_sizes.begin()+step_i, new_step_size);
#      step_energies.insert(step_energies.begin()+step_i, 1.0e100);
#      if ( step_i <= best_step )
#        best_step++;
#
#      // compute new value
#      if (formic::mpi::rank() == 0) {
#        formic::xcopy(n, x0, 1, x1, 1);
#        formic::xaxpy(n, step_sizes.at(step_i), p, 1, x1, 1);
#      }
#      worker.get_value(n, x1, step_energies.at(step_i));
#      if (formic::mpi::rank() == 0) {
#        formic::of << boost::format("for a step size of %16.8e") % step_sizes.at(step_i);
#        formic::of << boost::format("      val = %20.12f") % step_energies.at(step_i) << std::endl << std::endl;
#      }
#
#      // if this was the best step size so far, save it
#      if ( step_energies.at(step_i) < best_step_energy ) {
#        best_step = step_i;
#        best_step_size = step_sizes.at(step_i);
#        best_step_energy = step_energies.at(step_i);
#      }
#
#      // if we have done two minima refinements, stop searching
#      if (n_refine > 1)
#        break;
#
#      //// if the step sizes have become too close together, stop searching
#      //const double step_thresh = 1.0e-6;
#      //if ( best_step < step_sizes.size()-1 ) {
#      //  if ( std::abs( ( step_sizes.at(best_step) - step_sizes.at(best_step+1) ) / step_sizes.at(best_step+1) ) < step_thresh )
#      //    break;
#      //}
#      //if ( best_step > 0 ) {
#      //  if ( std::abs( ( step_sizes.at(best_step) - step_sizes.at(best_step-1) ) / step_sizes.at(best_step) ) < step_thresh )
#      //    break;
#      //}
#
#    }
#
#    if (formic::mpi::rank() == 0) {
#
#      // if a zero step was best, do not apply any update
#      if ( best_step == 0 ) {
#
#        formic::xcopy(n, x0, 1, x1, 1);
#
#        formic::of << "The line search failed to find a step that lowers the energy.  No update was applied." << std::endl << std::endl;
#
#      // otherwise, apply the update corresponding to the best step size
#      } else {
#
#        formic::xcopy(n, x0, 1, x1, 1);
#        formic::xaxpy(n, best_step_size, p, 1, x1, 1);
#
#        if (formic::mpi::rank() == 0)
#          formic::of << boost::format("The line search's best step size was %12.6f with a value of %20.12f") % best_step_size % best_step_energy
#                     << std::endl
#                     << boost::format("The corresponding update has been applied.")
#                     << std::endl
#                     << std::endl;
#
#      }
#
#    }
#
#    //// stop the timer
#    //formic::stop_timer("lbgfs loose line search");
#
#    // return the step size used
#    step_arg = best_step_size;
#
#  }
#
#  template <class T> inline void lbfgs_shift_down(std::vector<T> & vec) {
#      T temp = vec[0];
#      for (int i = 0; i < vec.size()-1; i++)
#        vec[i] = vec[i+1];
#      vec[vec.size()-1] = temp;
#  }

# shifts the current history back by one
def lbfgs_shift_down(x):
  if len(x.shape) == 2:
    temp = 1.0 * x[:,0]
    for i in range(x.shape[1]-1):
      x[:,i] = 1.0 * x[:,i+1]
    x[:,x.shape[1]-1] = 1.0 * temp
  elif len(x.shape) == 1:
    temp = 1.0 * x[0]
    #print ('temp = 1.0 * x[0] = ',temp)
    for i in range(x.shape[0]-1):
      #print ('for index ',i,' of the vector...')
      #print ('x[i] = 1.0 * x[i+1] = ',x[i+1])
      x[i] = 1.0 * x[i+1]
    x[x.shape[0]-1] = 1.0 * temp
    #print ('x[19] = 1.0 * temp = ', temp)
    #print ('final x = \n',x)
  else:
    raise RuntimeError("lbfgs_shift_down received x for which len(x.shape) is %i" % len(x.shape))

# manage the evaluation of the function, gradient, and possible inverse hessian approximation
def lbfgs_do_internal_evaluation(evaluation_function, x, original_shape):

  # get number of variables
  n = 0 + x.size

  # perform the evaluation
  #print ((1.0 * np.reshape(1.0 * x, original_shape)).shape)
  result = evaluation_function(1.0 * np.reshape(1.0 * x, original_shape))

  #print ('original shape: ',original_shape)
  #print ('x shape: ',x.shape)
  #print ('x: ',x)

  # if we don't get the right number of results, there is a problem
  if len(result) < 2 or len(result) > 3:
    raise RuntimeError("evaluation_function given to lbfgs should return two or three arguments:  value, gradient, and (optionally) hessian inverse function")

  # if we don't have the approximate inverse hessian, use the identity
  if len(result) < 3:
    hess_inv = lambda y: 1.0 * np.reshape(y, [n,1])

  # if we do have the approximate inverse hessian function, use it
  else:
    hess_inv = lambda y: 1.0 * np.reshape( result[2]( 1.0 * np.reshape(y, original_shape) ), [n,1])

  # ensure gradient shape is the column vector shape we are using internally
  g = 1.0 * np.reshape(result[1], [n,1])

  # return the value, the gradient, and the approximate hessian function
  return result[0], g, hess_inv


def lbfgs(evaluation_function,
          x0,
          max_iter=100,
          max_hist=20,
          grad_thresh=1.0e-5,
          value_thresh=-1.0,
          step_control_func=None):
  """ Performs an L-BFGS minimization.

      evaluation_function: takes the current variables as its sole argument
                           and returns the function value, the gradient, and
                           (optionally) a function that operates by an
                           approximation of the hessian inverse.
                           Usage:  result = evaluation_function(x)
                           where len(result) is either 2 or 3
                           The usage of the approximate hessian inverse is:
                             step = hess_inv(grad)
                           Of course, L-BFGS will further modify this step
                           using the history of gradients and function values.

      x0: the initial variable values to start the minimization from

      step_control_func: Function that takes as arguments the current function value,
                         the current gradient, the current variable values, and the
                         proposed step and returns an appropriately modified step.
                         Could involve line searching, applying a trust radius,
                         scaling the step, or perhaps returning the step unaltered.

  """

  ## print greeting
  #print("\n################################")
  #print(  "# Starting L-BFGS Minimization #")
  #print(  "################################\n")

  # get number of variables
  n = 0 + x0.size

  # get reshaping of x0 into shape used internally here
  x0i = 1.0 * np.reshape(1.0 * x0, [n,1])

  # choose how many previous gradients to remember
  m = 0 + max_hist

  # allocate storage for the gradient and position differences
  grad_diffs = np.zeros([n,m])
  pos_diffs  = np.zeros([n,m])

  # allocate search direction
  p = np.zeros([n,1])

  # allocate q vector
  q = np.zeros([n,1])

  # allocate z vector
  z = np.zeros([n,1])

  # allocate r and a vectors
  r_vec = np.zeros([m])
  a_vec = np.zeros([m])

  # evaluate initial value, gradient, and approximate inverse hessian function
  f0, g0, hess_inv = lbfgs_do_internal_evaluation(evaluation_function, x0i, x0.shape)

  #print("initial position:\n")
  #for i in range(n):
  #  print("%20.12f" % x0i[i,0])
  #print("\ninitial gradient:\n")
  #for i in range(n):
  #  print("%20.12f" % g0[i,0])
  #print("")

  # compute and print the norm of the initial gradient
  print("initial function value = %20.12f" % f0)
  grad_norm = np.linalg.norm(g0)
  print("initial gradient norm  = %20.6e\n" % grad_norm)

  f1s = []

  # iterate
  converged = False;
  m_used = 0;
  for ii in range(max_iter):

    ## print a greeting
    print("L-BFGS:  starting iteration %4i\n" % ii)

    # compute the search direction ( p = - h * g0 )
    #print ('COMPUTING SEARCH DIRECTION...')
    #print ('m ',m)
    if m > 0:	# if use of history is allowed
      q.fill(0.0)
      q += g0
      #qbh = np.zeros([n,1])
      #qbh += g0
      #print ('\nr_vec = \n',r_vec)
      #print ('\na_vec = \n',a_vec)
      for i in range(m-1, m - m_used - 1, -1):	# range over previous history (last to first, decreasing)
        #print ('range over previous history: ',i)
        #print ('pos_diffs[:,i] = \n',pos_diffs[:,i])
        #print ('q[:,0] = \n',q[:,0])
        #print ('r_vec[i] = \n',r_vec[i])
        a_vec[i] = r_vec[i] * np.sum( pos_diffs[:,i] * q[:,0] )
        #print ('a_vec after update = \n',a_vec)
        #print ('a_vec[i] = \n',a_vec[i])
        #print ('grad_diffs[:,i:i+1] = \n',grad_diffs[:,i:i+1])
        q -= a_vec[i] * grad_diffs[:,i:i+1]

        #print ('\n\nby hand...q = \n')
        #qbh -= np.sum( pos_diffs[:,i] * g0 * grad_diffs[:,i] ) / np.sum( grad_diffs[:,i] * pos_diffs[:,i] )
        #print (qbh)
        #print ('\n\n')


      #print ('q = new grad = \n',q)

      #zbh = np.zeros([n,1])
      #zbh += hess_inv(qbh)
      #print('zbh = hess.qbh = \n',zbh)

      z.fill(0.0)
      z += hess_inv(q)
      #print ('z = hess.q = \n',z)
      for i in range(m - m_used, m):	# range over previous history (first to last, inceasing)
        #print ('range over something i dont know yet: ',i)
        b = r_vec[i] * np.sum( grad_diffs[:,i] * z[:,0] )
        z += ( a_vec[i] - b ) * pos_diffs[:,i:i+1]
      p.fill(0.0)
      p -= z
    else:
      #print ('is this ever touched????????')
      p.fill(0.0)
      p -= hess_inv(g0)

    #if ii == 1: exit()

    # apply step control (might involve line searching, scaling, etc.)
    if step_control_func is not None:
      p = 1.0 * np.reshape( step_control_func(f0,
                                              1.0 * np.reshape( g0, x0.shape),
                                              1.0 * np.reshape(x0i, x0.shape),
                                              1.0 * np.reshape(  p, x0.shape),
                                              evaluation_function), [n,1] )

    # take the step
    x1 = x0i + p
    #print("took step of size %.4e\n" % np.linalg.norm(p))

    # evaluate new value, gradient, and approximate inverse hessian function
    f1, g1, hess_inv = lbfgs_do_internal_evaluation(evaluation_function, x1, x0.shape)

    # compute and print the norm of the new gradient
    grad_norm = np.linalg.norm(g1)
    #print(" function val = %20.12f" % f1)
    #print("gradient norm = %20.12e" % grad_norm)
    #print("gradient: \n",g1)
    #print(" pos_diffs norm = %20.12e" % np.linalg.norm(pos_diffs))
    #print("grad_diffs norm = %20.12e" % np.linalg.norm(grad_diffs))
    #print("Func value diff = %20.12e" % np.abs(f1-f0))

    #f1s.append(f1)
    #print ('func values so far:\n',np.reshape(f1s,[-1,1]))

    #print("\nnew position:\n")
    #for i in range(n):
    #  print("%20.12f" % x1[i,0])
    #print("\nnew gradient:\n")
    #for i in range(n):
    #  print("%20.12f" % g1[i,0])
    #print("")

    # TODO
    # if the gradient or energy difference is small enough, stop iterating
    converged = ( grad_norm < grad_thresh ) or ( np.abs(f1-f0) < value_thresh )
    #converged = ( grad_norm < grad_thresh )
    #if ii > 20:
    #  converged = ( np.linalg.norm(pos_diffs) < 1e-10) or ( np.linalg.norm(grad_diffs) < 1e-6 )
    if converged:
      break

    # update the history of gradient differences, position differences, and r values
    if m > 0:

      lbfgs_shift_down(pos_diffs)
      pos_diffs[:,m-1] = np.reshape( x1 - x0i, [-1] )

      lbfgs_shift_down(grad_diffs)
      grad_diffs[:,m-1] = np.reshape( g1 - g0, [-1] )

      #print ('r_vec before shift: \n',r_vec)
      lbfgs_shift_down(r_vec)
      #print ('r_vec after shift: \n',r_vec)
      #print ('grad_diffs[:,m-1] = \n',grad_diffs[:,m-1])
      #print ('pos_diffs[:,m-1] = \n',pos_diffs[:,m-1])
      #print ('sum = ', np.sum(grad_diffs[:,m-1] * pos_diffs[:,m-1]))
      #print ('dot = ', np.dot(grad_diffs[:,m-1].T, pos_diffs[:,m-1]))
      r_vec[m-1] = 1.0 / np.sum( grad_diffs[:,m-1] * pos_diffs[:,m-1] )
      #print ('r_vec after update: \n',r_vec)

      #exit()

      m_used = min(m, m_used + 1)

    # move the new gradient, position, and function value into the old gradient, position, and function value
    f0 = f1
    x0i = x1
    g0 = g1

  # end of iterations over ii

  # check that the method converged
  if converged:
    print("L-BFGS method converged!")
  else:
    print("L-BFGS method did not converge...")
  print('final macroiteration parameters: \n',np.reshape(x0i,[1,-1]))
  return x0i, grad_norm


# a function to minimize that I made up for testing L-BFGS
def eval_func_for_testing_lbfgs(xi):

  a = 0.05
  b = 0.5
  c = 0.1

  n = xi.size
  x = np.reshape(xi, [-1])
  g = np.zeros([x.size])
  h = np.zeros([x.size])

  w = 0.0
  w -= b * np.sum( np.square(x) )
  for i in range(1,n):
    w -= c * ( x[i] - x[i-1] )**2.0
  w = np.exp(w)

  f = ( x[0] - a )**2.0 - w

  g[0] = 2.0 * ( x[0] - a ) - w * ( -2.0 * c * ( x[0] - x[1] ) - 2.0 * b * x[0] )
  for i in range(1,n-1):
    g[i] = -1.0 * w * ( 2.0 * c * ( x[i-1] - x[i] ) - 2.0 * c * ( x[i] - x[i+1] ) - 2.0 * b * x[i] )
  g[n-1] = -1.0 * w * ( 2.0 * c * ( x[n-2] - x[n-1] ) - 2.0 * b * x[n-1])

  h[0] = 2.0 + 2.0 * b + 2.0 * c
  for i in range(1,n-1):
    h[i] = 2.0 * b + 4.0 * c
  h[n-1] = 2.0 * b + 2.0 * c


  print ('test x shape: ',x.shape)
  print ('test hess shape: ',(np.reshape( np.reshape(x, [-1])/h,x.shape)).shape)
  approx_hess_inv = lambda x: np.reshape( np.reshape(x, [-1]) / h, x.shape)
  print (approx_hess_inv(x))

  # if we return just f and g, L-BFGS will use the identity operator for the inverse hessian
  #return f, g

  # instead, we can provide our own approximation for the inverse hessian
  return f, g, approx_hess_inv

# function that runs a test of the L-BFGS minimization
def run_test_of_lbfgs():
  x0 = np.array([-0.3, -0.07, 0.15])
  lbfgs(eval_func_for_testing_lbfgs, x0, max_iter=100, max_hist=10, grad_thresh=1.0e-10,
        step_control_func=step_control_for_testing_lbfgs)

#======================
# STEP CONTROL...
#======================

# extremely simple step control for testing L-BFGS
def step_control_for_testing_lbfgs_original(val, grad, x, p):
  max_elem = np.max(np.abs(p))
  if max_elem > 0.5:
    return ( 0.5 / max_elem ) * p * 0.1
  return p * 0.1



# extremely simple step control for testing L-BFGS
def step_control_frozenC(val, grad, x, p, eval_func):
  nstr = 36	#LiH noCore, H2O frozenCore(4e,4o), ccpVDZ
  Xlen = 60	#LiH noCore, ccpVDZ
  #Xlen = 68	#H2O frozenCore(4e,4o), ccpVDZ
  p_X = p[:Xlen]
  p_C = p[Xlen:]
  max_elem_C = np.max(np.abs(p_C))
  max_elem_X = np.max(np.abs(p_X))
  #if max_elem_C > 0.5:
  #  p_C = ( 0.5 / max_elem_C ) * p_C
  p_C_zeros = np.zeros(p_C.shape)
  if max_elem_X > 0.01:
    p_X = ( 0.01 / max_elem_X ) * p_X
  p_scaled = np.concatenate( [p_X, p_C_zeros], 0 )
  return p_scaled

# extremely simple step control for testing L-BFGS
def step_control_gridsearch(val, grad, x, p, evaluation_function):
  #nstr = 36	#LiH noCore, H2O frozenCore(4e,4o), ccpVDZ
  #Xlen = 60	#LiH noCore, ccpVDZ
  #Xlen = 68	#H2O frozenCore(4e,4o), ccpVDZ
  #p_X = p[:Xlen]
  #p_C = p[Xlen:]
  #max_elem_C = np.max(np.abs(p_C))
  #max_elem_X = np.max(np.abs(p_X))
  #if max_elem_C > 0.001:
  #  p_C = ( 0.001 / max_elem_C ) * p_C
  #if max_elem_X > 0.001:
  #  p_X = ( 0.001 / max_elem_X ) * p_X
  #p_scaled = np.concatenate( [p_X, p_C], 0 )
  func_vals = []
  i_list = []
  for i in np.arange(-0.5,0.5,5e-2):
    x_i = x + ( i * p )
    f_i, g, inv_hess = lbfgs_do_internal_evaluation(evaluation_function, x_i, x.shape)
    func_vals.append(f_i)
    i_list.append(i)
  #for funcval in func_vals:
    #print ('i = %2.4g   func value = %3.6g   diff = %3.6g' %(i_list[func_vals.index(funcval)], funcval, funcval-val))
    #print ('i = ', i_list[func_vals.index(funcval)], ' func value = ', funcval, 'diff = ', funcval-val)

  i_ind = func_vals.index(min(func_vals))
  i_fmin = i_list[i_ind]
  print ('accepted i = %2.4g   min func value = %3.6g  diff = %3.6g' %(i_fmin, min(func_vals), min(func_vals)-val))
  p_scaled = p * i_fmin
  #print ('p (unscaled):\n',p) 
  #print ('p_scaled:') 
  #print (p_scaled) 
  return p_scaled


#def backtrack_linesearch(x_vec, d_vec, tar_func, grad_func, step_func, step_args, \
def backtrack_linesearch(f0_val, g0_vec, x_vec, d_vec, evaluation_function, \
    max_iter=50, debug_print=False, num_tol=1e-15):
  alph = 0.5
  gamma = 1
  # constants
  iter_ind = 0
  c1 = 1e-4
  c2 = 0.9
  # evaluate function, grad at initial point and do first step
  #f0_val = tar_func(x_vec, *step_args)
  #g0_vec = grad_func(x_vec, *step_args)
  g0_proj = g0_vec @ d_vec # projected gradient along the linesearch direction
  # print out some stuff
  spacing = " "*26
  if debug_print:
    backtrack_header = "{} | backtrack linesearch\n{} | {:^12s} | {:^12s} | {:^12s} | {:^12s} | {:^12s}".format(\
        spacing, spacing, "gamma", "f_val", "g_proj", "wolfe func", "wolfe gproj")
    backtrack_iter_line = "{} | {:>12.4e} | {:>12.4e} | {:>12.4e} | {:>12.4e} | {:>12.4e}"
    print(backtrack_header)
    print(backtrack_iter_line.format(spacing, 0, f0_val, g0_proj, 0, 0))
  while iter_ind < max_iter:
    # make a temporary step and evaluate function and gradient
    s_vec = x_vec + gamma*d_vec
    f_val, g_vec, inv_hess_func = lbfgs_do_internal_evaluation(evaluation_function, s_vec, s_vec.shape)
    g_vec = np.reshape(g_vec,d_vec.shape)
    g_proj = g_vec @ d_vec
    # evaluate the wolfe conditions at the new position 
    wc1 = f_val - (f0_val + c1*gamma*g0_proj) # should be less than zero
    wc2 = c2*g0_proj - g_proj                 # should be less than zero 
    if debug_print:
      print(backtrack_iter_line.format(spacing, gamma, f_val, g_proj, wc1, wc2))
    # break if wolfe conditions are met (i.e. negative)
    if wc1 < num_tol and wc2 < num_tol:
      break
    # break if wolfe conditions are below numerical precision
    #if np.abs(wc1) < num_tol and np.abs(wc2) < num_tol:
    #  break
    # otherwise rescale the steplength by alph
    gamma = gamma*alph
    iter_ind = iter_ind + 1
  # rescale and make the step, calculate final values
  d_vec = gamma*d_vec
  #x_vec = x_vec + d_vec
  #f_val, g_vec, inv_hess_func = lbfgs_do_internal_evaluation(evaluation_function, x_vec, x_vec.shape)
  #return x_vec, d_vec, f_val, g_vec, step_args, iter_ind
  return d_vec

#}
#
#}
#
##endif
