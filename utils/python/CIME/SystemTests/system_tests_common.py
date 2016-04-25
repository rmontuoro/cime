"""
Base class for CIME system tests
"""
import shutil, glob
from CIME.XML.standard_module_setup import *
from CIME.case import Case
from CIME.XML.env_run import EnvRun
from CIME.utils import run_cmd
from CIME.case_setup import case_setup
import CIME.build as build

logger = logging.getLogger(__name__)

class SystemTestsCommon(object):
    def __init__(self, caseroot=os.getcwd(), case=None):
        """
        initialize a CIME system test object, if the file LockedFiles/env_run.orig.xml
        does not exist copy the current env_run.xml file.  If it does exist restore values
        changed in a previous run of the test.
        """
        print caseroot
        self._caseroot = caseroot
        # Needed for sh scripts
        os.environ["CASEROOT"] = caseroot
        if case is None:
            self._case = Case(caseroot)
        else:
            self._case = case

        if os.path.isfile(os.path.join(caseroot, "LockedFiles", "env_run.orig.xml")):
            self.compare_env_run()
        elif os.path.isfile(os.path.join(caseroot, "env_run.xml")):
            lockedfiles = os.path.join(caseroot, "Lockedfiles")
            try:
                os.stat(lockedfiles)
            except:
                os.mkdir(lockedfiles)
            shutil.copy("env_run.xml",
                        os.path.join(lockedfiles, "env_run.orig.xml"))

        self._case.set_initial_test_values()
        case_setup(self._caseroot, reset=True, test_mode=True)


    def build(self, sharedlib_only=False, model_only=False):
        build.case_build(self._caseroot, case=self._case,
                         sharedlib_only=sharedlib_only, model_only=model_only)


    def run(self):
        with open("TestStatus", 'a') as f:
            f.write("PEND %s RUN\n"%self._case.get_value("CASEBASEID"))

        with open("TestStatus", 'r') as f:
            teststatusfile = f.read()

        rc, out, err = run_cmd("./case.run", ok_to_fail=True)
        if rc == 0:
            result = "PASS"
        else:
            result = "FAIL"

        with open("TestStatus.log", 'a') as f:
            f.write("case.run output is %s"%out)
            f.write("case.run error is %s"%err)

        li = teststatusfile.rsplit('PEND', 1)
        teststatusfile = result.join(li)
        with open("TestStatus", 'w') as f:
            f.write(teststatusfile)


        return

    def report(self):
        newestcpllogfile = self._getlatestcpllog()
        if "SUCCESSFUL TERMINATION" in open(newestcpllogfile).read():
            with open("TestStatus", "a") as fd:
                fd.write("PASS %s : successful coupler log\n"%(self._case.get_value("CASEBASEID")))
        else:
            with open("TestStatus", "a") as fd:
                fd.write("FAIL %s : coupler log indicates a problem\n"%(self._case.get_value("CASEBASEID")))

        self._checkformemleak(newestcpllogfile)
        self._compare
        return

    def _checkformemleak(self, cpllog):
        """
        Examine memory usage as recorded in the cpl log file and look for unexpected
        increases.
        """


        cmd = os.path.join(self._case.get_value("SCRIPTSROOT"),"Tools","check_memory.pl")
        rc, out, err = run_cmd("%s -file1 %s -m 1.5"%(cmd, cpllog),ok_to_fail=True)
        if rc == 0:
            with open("TestStatus", "a") as fd:
                fd.write("PASS %s memleak\n"%(self._case.get_value("CASEBASEID")))
        else:
            with open(os.path.join(test_dir, "TestStatus.log"), "a") as fd:
                fd.write("memleak out: %s\n\nerror: %s"%(out,err))
            with open(os.path.join(test_dir, "TestStatus"), "a") as fd:
                fd.write("FAIL %s memleak\n"%(self._case.get_value("CASEBASEID")))

    def compare_env_run(self, expected=None):
        f1obj = EnvRun(self._caseroot, "env_run.xml")
        f2obj = EnvRun(self._caseroot, os.path.join("LockedFiles", "env_run.orig.xml"))
        diffs = f1obj.compare_xml(f2obj)
        for key in diffs.keys():
            if expected is not None and key in expected:
                logging.warn("  Resetting %s for test"%key)
                f1obj.set_value(key, f2obj.get_value(key, resolved=False))
            else:
                print "Found difference in %s: case: %s original value %s" %\
                    (key, diffs[key][0], diffs[key][1])
                print " Use option --force to run the test with this"\
                    " value or --reset to reset to original"
                return False
        return True

    def _getlatestcpllog(self):
        """
        find and return the latest cpl log file in the run directory
        """
        cpllog = min(glob.iglob(os.path.join(
                    self._case.get_value('RUNDIR'),'cpl.log.*')), key=os.path.getctime)
        return cpllog



    def _compare(self):
        """
        check to see if there are history files to be compared, compare if they are there
        """
        cmd = os.path.join(self._case.get_value("SCRIPTSROOT"),"Tools",
                                                "component_compare_test.sh")
        rc, out, err = run_cmd("%s -rundir %s -testcase %s -testcase_base %s -suffix1 base -suffix2 rest"
                               %(cmd, self._case.get_value('RUNDIR'), self._case.get_value('CASE'),
                                 self._case.get_value('CASEBASEID')), ok_to_fail=True)
        if rc == 0:
            with open("TestStatus", "a") as fd:
                fd.write(out+"\n")
        else:
            with open("TestStatus.log", "a") as fd:
                fd.write("Component_compare_test.sh failed out: %s\n\nerr: %s\n"%(out,err))

    def compare_baseline(self):
        """
        compare the current test output to a baseline result
        """
        baselineroot = self._case.get_value("BASELINE_ROOT")
        test_dir = self._case.get_value("CASEROOT")
        basecmp_dir = os.path.join(baselineroot, self._case.get_value("BASECMP_CASE"))
        for bdir in (baselineroot, basecmp_dir):
            if not os.path.isdir(bdir):
                with open(os.path.join(test_dir, "TestStatus"), "a") as fd:
                    fd.write("GFAIL %s baseline\n",self._case.get_value("CASEBASEID"))
                with open(os.path.join(test_dir, "TestStatus.log"), "a") as fd:
                    fd.write("ERROR %s does not exist",bdir)
                return -1
        compgen = os.path.join(self._case.get_value("SCRIPTSROOT"),"Tools",
                               "component_compgen_baseline.sh")
        compgen += " -baseline_dir "+basecmp_dir
        compgen += " -test_dir "+self._case.get_value("RUNDIR")
        compgen += " -compare_tag "+self._case.get_value("BASELINE_NAME_CMP")
        compgen += " -testcase "+self._case.get_value("CASE")
        compgen += " -testcase_base "+self._case.get_value("CASEBASEID")
        rc, out, err = run_cmd(compgen, ok_to_fail=True)
        with open(os.path.join(test_dir, "TestStatus"), "a") as fd:
            fd.write(out)
        if rc != 0:
            with open(os.path.join(test_dir, "TestStatus.log"), "a") as fd:
                fd.write("Error in Baseline compare: %s"%err)

    def generate_baseline(self):
        """
        generate a new baseline case based on the current test
        """
        newestcpllogfile = self._getlatestcpllog()
        baselineroot = self._case.get_value("BASELINE_ROOT")
        basegen_dir = os.path.join(baselineroot, self._case.get_value("BASEGEN_CASE"))
        test_dir = self._case.get_value("CASEROOT")
        for bdir in (baselineroot, basegen_dir):
            if not os.path.isdir(bdir):
                with open(os.path.join(test_dir, "TestStatus"), "a") as fd:
                    fd.write("GFAIL %s baseline\n" % self._case.get_value("CASEBASEID"))
                with open(os.path.join(test_dir, "TestStatus.log"), "a") as fd:
                    fd.write("ERROR %s does not exist" % bdir)
                return -1
        compgen = os.path.join(self._case.get_value("SCRIPTSROOT"),"Tools",
                               "component_compgen_baseline.sh")
        compgen += " -baseline_dir "+basegen_dir
        compgen += " -test_dir "+self._case.get_value("RUNDIR")
        compgen += " -generate_tag "+self._case.get_value("BASELINE_NAME_GEN")
        compgen += " -testcase "+self._case.get_value("CASE")
        compgen += " -testcase_base "+self._case.get_value("CASEBASEID")
        rc, out, err = run_cmd(compgen, ok_to_fail=True)
        # copy latest cpl log to baseline
        shutil.copyfile(newestcpllogfile,
                        os.path.join(basegen_dir,
                                     os.path.basename(newestcpllogfile)))

        with open(os.path.join(test_dir, "TestStatus"), "a") as fd:
            fd.write(out+"\n")
        if rc != 0:
            with open(os.path.join(test_dir, "TestStatus.log"), "a") as fd:
                fd.write("Error in Baseline Generate: %s"%err)



class FakeTest(SystemTestsCommon):

    def fake_build(self, script, sharedlib_only=False, model_only=False):
        if (not sharedlib_only):
            exeroot = self._case.get_value("EXEROOT")
            cime_model = self._case.get_value("MODEL")
            modelexe = os.path.join(exeroot, "%s.exe"%cime_model)

            with open(modelexe, 'w') as f:
                f.write("#!/bin/bash\n")
                f.write(script)

            os.chmod(modelexe, 0755)
            self._case.set_value("BUILD_COMPLETE", True)
            self._case.flush()

class TESTRUNPASS(FakeTest):

    def build(self, sharedlib_only=False, model_only=False):
        rundir = self._case.get_value("RUNDIR")
        script = \
"""
echo Insta pass
echo SUCCESSFUL TERMINATION > %s/cpl.log.$LID
""" % rundir
        self.fake_build(script,
                        sharedlib_only=sharedlib_only, model_only=model_only)

class TESTRUNDIFF(FakeTest):

    def build(self, sharedlib_only=False, model_only=False):
        rundir = self._case.get_value("RUNDIR")
        cimeroot = self._case.get_value("CIMEROOT")
        case = self._case.get_value("CASE")
        script = \
"""
echo Insta pass
echo SUCCESSFUL TERMINATION > %s/cpl.log.$LID
cp %s/utils/python/tests/cpl.hi1.nc.test %s/%s.cpl.hi.0.nc.base
""" % (rundir, cimeroot, rundir, case)
        self.fake_build(script,
                        sharedlib_only=sharedlib_only, model_only=model_only)

class TESTRUNFAIL(FakeTest):

    def build(self, sharedlib_only=False, model_only=False):
        rundir = self._case.get_value("RUNDIR")
        script = \
"""
echo Insta fail
echo model failed > %s/cpl.log.$LID
exit -1
""" % rundir
        self.fake_build(script,
                        sharedlib_only=sharedlib_only, model_only=model_only)

class TESTBUILDFAIL(FakeTest):

    def build(self, sharedlib_only=False, model_only=False):
        if (not sharedlib_only):
            expect(False, "ERROR: Intentional fail for testing infrastructure")

class TESTRUNSLOWPASS(FakeTest):

    def build(self, sharedlib_only=False, model_only=False):
        rundir = self._case.get_value("RUNDIR")
        script = \
"""
sleep 300
echo Slow pass
echo SUCCESSFUL TERMINATION > %s/cpl.log.$LID
""" % rundir
        self.fake_build(script,
                        sharedlib_only=sharedlib_only, model_only=model_only)
