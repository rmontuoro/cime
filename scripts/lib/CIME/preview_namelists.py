"""
API for preview namelist
"""

from CIME.XML.standard_module_setup import *

import glob, shutil, imp
logger = logging.getLogger(__name__)

def create_dirs(case):
    """
    Make necessary directories for case
    """
    # Get data from XML
    exeroot  = case.get_value("EXEROOT")
    libroot  = case.get_value("LIBROOT")
    incroot  = case.get_value("INCROOT")
    rundir   = case.get_value("RUNDIR")
    caseroot = case.get_value("CASEROOT")

    docdir = os.path.join(caseroot, "CaseDocs")
    dirs_to_make = []
    models = case.get_values("COMP_CLASSES")
    for model in models:
        dirname = model.lower()
        dirs_to_make.append(os.path.join(exeroot, dirname, "obj"))

    dirs_to_make.extend([exeroot, libroot, incroot, rundir, docdir])

    for dir_to_make in dirs_to_make:
        if (not os.path.isdir(dir_to_make)):
            try:
                logger.debug("Making dir '%s'" % dir_to_make)
                os.makedirs(dir_to_make)
            except OSError as e:
                expect(False, "Could not make directory '%s', error: %s" % (dir_to_make, e))

    # As a convenience write the location of the case directory in the bld and run directories
    for dir_ in (exeroot, rundir):
        with open(os.path.join(dir_,"CASEROOT"),"w+") as fd:
            fd.write(caseroot+"\n")

def create_namelists(case, component=None):
    """
    Create component namelists
    """
    case.flush()

    create_dirs(case)

    casebuild = case.get_value("CASEBUILD")
    caseroot = case.get_value("CASEROOT")
    rundir = case.get_value("RUNDIR")

    docdir = os.path.join(caseroot, "CaseDocs")

    # Load modules
    case.load_env()

    logger.info("Creating component namelists")

    # Create namelists - must have cpl last in the list below
    # Note - cpl must be last in the loop below so that in generating its namelist,
    # it can use xml vars potentially set by other component's buildnml scripts
    xmlfac = {}
    cpl_ninst = case.get_value("NINST_CPL")
    models = case.get_values("COMP_CLASSES")
    models += [models.pop(0)]
    for model in models:
        model_str = model.lower()
        config_file = case.get_value("CONFIG_%s_FILE" % model_str.upper())
        config_dir = os.path.dirname(config_file)
        if model_str == "cpl":
            compname = "drv"
            complist = [m for m in models if m.upper() != "CPL"]
            if cpl_ninst > 1:
                xmlfac = {"NINST" : cpl_ninst, "NTASKS" : 1}
        else:
            compname = case.get_value("COMP_%s" % model_str.upper())
            complist = [model_str.upper()]
            if cpl_ninst > 1:
                xmlfac = {"NINST" : cpl_ninst, "NTASKS" : cpl_ninst}

        xmlsave = {}
        for k in xmlfac.keys():
            for m in complist:
                key = "%s_%s" % (k, m.upper())
                xmlsave[key] = case.get_value("%s" % key)

        if component is None or component == model_str:
            cmd = os.path.join(config_dir, "buildnml")
            do_run_cmd = False
            # This code will try to import and run each buildnml as a subroutine
            # if that fails it will run it as a program in a seperate shell
            try:
                with open(cmd, 'r') as f:
                    first_line = f.readline()
                if "python" in first_line:
                    mod = imp.load_source("buildnml", cmd)
                    logger.info("   Calling %s buildnml"%compname)
                    for key, value in xmlsave.items():
                        case.set_value("%s" % key, xmlfac[key.split('_')[0]] * value)
                    mod.buildnml(case, caseroot, compname)
                    for key, value in xmlsave.items():
                        case.set_value("%s" % key, value)
                    case.flush()
                else:
                    raise SyntaxError
            except SyntaxError as detail:
                if 'python' in first_line:
                    expect(False, detail)
                else:
                    do_run_cmd = True
            except AttributeError:
                do_run_cmd = True
            except:
                raise

            if do_run_cmd:
                logger.info("   Running %s buildnml"%compname)
                for key, value in xmlsave.items():
                    case.set_value("%s" % key, xmlfac[key.split('_')[0]] * value)
                case.flush()
                output = run_cmd_no_fail("%s %s" % (cmd, caseroot), verbose=False)
                logger.info(output)
                for key, value in xmlsave.items():
                    case.set_value("%s" % key, value)
                case.flush()
                # refresh case xml object from file
                case.read_xml()

    logger.info("Finished creating component namelists")

    # Save namelists to docdir
    if (not os.path.isdir(docdir)):
        os.makedirs(docdir)
        try:
            with open(os.path.join(docdir, "README"), "w") as fd:
                fd.write(" CESM Resolved Namelist Files\n   For documentation only DO NOT MODIFY\n")
        except (OSError, IOError) as e:
            expect(False, "Failed to write %s/README: %s" % (docdir, e))

    for cpglob in ["*_in_[0-9]*", "*modelio*", "*_in",
                   "*streams*txt*", "*stxt", "*maps.rc", "*cism.config*"]:
        for file_to_copy in glob.glob(os.path.join(rundir, cpglob)):
            logger.debug("Copy file from '%s' to '%s'" % (file_to_copy, docdir))
            shutil.copy2(file_to_copy, docdir)

    # Copy over chemistry mechanism docs if they exist
    if (os.path.isdir(os.path.join(casebuild, "camconf"))):
        for file_to_copy in glob.glob(os.path.join(casebuild, "camconf", "*chem_mech*")):
            shutil.copy2(file_to_copy, docdir)
