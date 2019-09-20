#include <iostream>
#include <vector>
#include "run_until.h"
#include "domain.h"
#include "output.h"
#include "input.h"
#include "update.h"
#include "scheme.h"
#include "var.h"

/* ---------------------------------------------------------------------- */

RunUntil::RunUntil(MPM *mpm) : Pointers(mpm) {}

/* ---------------------------------------------------------------------- */

Var RunUntil::command(vector<string> args)
{
  // cout << "In RunUntil::command()" << endl;

  if (args.size() < 1) {
    cout << "Illegal run command" << endl;
    exit(1);
  }

  mpm->init();

  update->scheme->setup();

  Var condition = input->parsev(args[0]);
  update->nsteps = INT_MAX;
  update->maxtime = -1;
  update->firststep = update->ntimestep;
  update->laststep = INT_MAX;
  update->scheme->run(Var("!("+condition.str()+")", !condition.result(mpm)));

  return Var(0);
}