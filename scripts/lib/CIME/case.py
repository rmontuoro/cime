"""
Wrapper around all env XML for a case.

All interaction with and between the module files in XML/ takes place
through the Case module.
"""
from copy import deepcopy
import glob, os, shutil, math, string
from CIME.XML.standard_module_setup import *

from CIME.utils                     import expect, get_cime_root, append_status
from CIME.utils                     import convert_to_type, get_model, get_project
from CIME.utils                     import get_current_commit
from CIME.check_lockedfiles         import LOCKED_DIR, lock_file
from CIME.XML.machines              import Machines
from CIME.XML.pes                   import Pes
from CIME.XML.files                 import Files
from CIME.XML.testlist              import Testlist
from CIME.XML.component             import Component
from CIME.XML.compsets              import Compsets
from CIME.XML.grids                 import Grids
from CIME.XML.batch                 import Batch
from CIME.XML.pio                   import PIO
from CIME.XML.archive               import Archive
from CIME.XML.env_test              import EnvTest
from CIME.XML.env_mach_specific     import EnvMachSpecific
from CIME.XML.env_case              import EnvCase
from CIME.XML.env_mach_pes          import EnvMachPes
from CIME.XML.env_build             import EnvBuild
from CIME.XML.env_run               import EnvRun
from CIME.XML.env_archive           import EnvArchive
from CIME.XML.env_batch             import EnvBatch
from CIME.XML.generic_xml           import GenericXML
from CIME.user_mod_support          import apply_user_mods
from CIME.case_setup import case_setup
from CIME.aprun import get_aprun_cmd_for_case

logger = logging.getLogger(__name__)

class Case(object):
    """
    https://github.com/ESMCI/cime/wiki/Developers-Introduction
    The Case class is the heart of the CIME Case Control system.  All
    interactions with a Case take part through this class.  All of the
    variables used to create and manipulate a case are defined in xml
    files and for every xml file there is a python class to interact
    with that file.

    XML files which are part of the CIME distribution and are meant to
    be readonly with respect to a case are typically named
    config_something.xml and the corresponding python Class is
    Something and can be found in file CIME.XML.something.py.  I'll
    refer to these as the CIME config classes.

    XML files which are part of a case and thus are read/write to a
    case are typically named env_whatever.xml and the cooresponding
    python modules are CIME.XML.env_whatever.py and classes are
    EnvWhatever.  I'll refer to these as the Case env classes.

    The Case Class includes an array of the Case env classes, in the
    configure function and it's supporting functions defined below
    the case object creates and manipulates the Case env classes
    by reading and interpreting the CIME config classes.

    """
    def __init__(self, case_root=None, read_only=True):

        if case_root is None:
            case_root = os.getcwd()
        self._caseroot = case_root
        logger.debug("Initializing Case.")
        self._env_files_that_need_rewrite = set()
        self._read_only_mode = True
        self._force_read_only = read_only
        self._primary_component = None

        self._env_entryid_files = []
        self._env_generic_files = []
        self._files = []

        self.read_xml()

        # Hold arbitary values. In create_newcase we may set values
        # for xml files that haven't been created yet. We need a place
        # to store them until we are ready to create the file. At file
        # creation we get the values for those fields from this lookup
        # table and then remove the entry.
        self.lookups = {}
        self.set_lookup_value('CIMEROOT',os.path.abspath(get_cime_root()))
        self._cime_model = get_model()
        self.set_lookup_value('MODEL', self._cime_model)
        self._compsetname = None
        self._gridname = None
        self._compsetsfile = None
        self._pesfile = None
        self._gridfile = None
        self._components = []
        self._component_classes = []
        self._component_description = {}
        self._is_env_loaded = False
        # these are user_mods as defined in the compset
        # Command Line user_mods are handled seperately
        self.thread_count = None
        self.total_tasks = None
        self.tasks_per_node = None
        self.num_nodes = None
        self.spare_nodes = None
        self.tasks_per_numa = None
        self.cores_per_task = None
        # check if case has been configured and if so initialize derived
        if self.get_value("CASEROOT") is not None:
            self.initialize_derived_attributes()

    def check_if_comp_var(self, vid):
        vid = vid
        comp = None
        iscompvar = False
        for env_file in self._env_entryid_files:
            vid, comp, iscompvar = env_file.check_if_comp_var(vid)
            if iscompvar:
                return vid, comp, iscompvar
        return vid, comp, iscompvar

    def initialize_derived_attributes(self):
        """
        These are derived variables which can be used in the config_* files
        for variable substitution using the {{ var }} syntax
        """
        env_mach_pes  = self.get_env("mach_pes")
        env_mach_spec = self.get_env('mach_specific')
        comp_classes  = self.get_values("COMP_CLASSES")
        pes_per_node  = self.get_value("PES_PER_NODE")

        self.total_tasks = env_mach_pes.get_total_tasks(comp_classes)
        self.thread_count = env_mach_pes.get_max_thread_count(comp_classes)
        self.tasks_per_node = env_mach_pes.get_tasks_per_node(self.total_tasks, self.thread_count)
        logger.debug("total_tasks %s thread_count %s"%(self.total_tasks, self.thread_count))

        self.tasks_per_numa = int(math.ceil(self.tasks_per_node / 2.0))
        smt_factor = max(1,int(self.get_value("MAX_TASKS_PER_NODE") / pes_per_node))

        threads_per_node = self.tasks_per_node * self.thread_count
        threads_per_core = 1 if (threads_per_node <= pes_per_node) else smt_factor
        self.cores_per_task = self.thread_count / threads_per_core

        mpi_attribs = {
            "compiler" : self.get_value("COMPILER"),
            "mpilib"   : self.get_value("MPILIB"),
            "threaded" : self.get_build_threaded(),
            }

        executable = env_mach_spec.get_mpirun(self, mpi_attribs, job="case.run", exe_only=True)[0]
        if executable is not None and "aprun" in executable:
            self.num_nodes = get_aprun_cmd_for_case(self, "acme.exe")[1]
            self.spare_nodes = env_mach_pes.get_spare_nodes(self.num_nodes)
            self.num_nodes += self.spare_nodes
        else:
            self.num_nodes, self.spare_nodes = env_mach_pes.get_total_nodes(self.total_tasks, self.thread_count)
            self.num_nodes += self.spare_nodes

    # Define __enter__ and __exit__ so that we can use this as a context manager
    # and force a flush on exit.
    def __enter__(self):
        if not self._force_read_only:
            self._read_only_mode = False
        return self

    def __exit__(self, *_):
        self.flush()
        self._read_only_mode = True
        return False

    def schedule_rewrite(self, env_file):
        assert not self._read_only_mode, \
            "case.py scripts error: attempted to modify an env file while in " \
            "read-only mode"
        self._env_files_that_need_rewrite.add(env_file)

    def read_xml(self):
        if(len(self._env_files_that_need_rewrite)>0):
            files = ""
            for env_file in self._env_files_that_need_rewrite:
                files += " "+env_file.filename
            expect(False,"Object(s) %s seem to have newer data than the corresponding case file"%files)

        self._env_entryid_files = []
        self._env_entryid_files.append(EnvCase(self._caseroot, components=None))
        components = self._env_entryid_files[0].get_values("COMP_CLASSES")
        self._env_entryid_files.append(EnvRun(self._caseroot, components=components))
        self._env_entryid_files.append(EnvBuild(self._caseroot, components=components))
        self._env_entryid_files.append(EnvMachPes(self._caseroot, components=components))
        self._env_entryid_files.append(EnvBatch(self._caseroot))
        if os.path.isfile(os.path.join(self._caseroot,"env_test.xml")):
            self._env_entryid_files.append(EnvTest(self._caseroot, components=components))
        self._env_generic_files = []
        self._env_generic_files.append(EnvMachSpecific(self._caseroot))
        self._env_generic_files.append(EnvArchive(self._caseroot))
        self._files = self._env_entryid_files + self._env_generic_files

    def get_case_root(self):
        """Returns the root directory for this case."""
        return self._caseroot

    def get_env(self, short_name, allow_missing=False):
        full_name = "env_%s.xml" % (short_name)
        for env_file in self._files:
            if os.path.basename(env_file.filename) == full_name:
                return env_file
        if allow_missing:
            return None
        expect(False,"Could not find object for %s in case"%full_name)

    def copy(self, newcasename, newcaseroot, newcimeroot=None, newsrcroot=None):
        newcase = deepcopy(self)
        for env_file in newcase._files: # pylint: disable=protected-access
            basename = os.path.basename(env_file.filename)
            env_file.filename = os.path.join(newcaseroot,basename)

        if newcimeroot is not None:
            newcase.set_value("CIMEROOT", newcimeroot)

        if newsrcroot is not None:
            newcase.set_value("SRCROOT", newsrcroot)

        newcase.set_value("CASE",newcasename)
        newcase.set_value("CASEROOT",newcaseroot)
        newcase.set_value("CONTINUE_RUN","FALSE")
        newcase.set_value("RESUBMIT",0)
        return newcase

    def flush(self, flushall=False):
        if not os.path.isdir(self._caseroot):
            # do not flush if caseroot wasnt created
            return
        if flushall:
            for env_file in self._files:
                self.schedule_rewrite(env_file)
        for env_file in self._env_files_that_need_rewrite:
            env_file.write()
        self._env_files_that_need_rewrite = set()

    def get_values(self, item, attribute=None, resolved=True, subgroup=None):
        results = []
        for env_file in self._env_entryid_files:
            # Wait and resolve in self rather than in env_file
            results = env_file.get_values(item, attribute, resolved=False, subgroup=subgroup)
            if len(results) > 0:
                new_results = []
                vtype = env_file.get_type_info(item)
                if resolved:
                    for result in results:
                        if type(result) is str:
                            result = self.get_resolved_value(result)
                            new_results.append(convert_to_type(result, vtype, item))
                        else:
                            new_results.append(result)
                else:
                    new_results = results
                return new_results

        for env_file in self._env_generic_files:
            results = env_file.get_values(item, attribute, resolved=False, subgroup=subgroup)
            if len(results) > 0:
                if resolved:
                    for result in results:
                        if type(result) is str:
                            new_results.append(self.get_resolved_value(result))
                        else:
                            new_results.append(result)
                else:
                    new_results = results
                return new_results
        # Return empty result
        return results

    def get_value(self, item, attribute=None, resolved=True, subgroup=None):
        result = None
        for env_file in self._env_entryid_files:
            # Wait and resolve in self rather than in env_file
            result = env_file.get_value(item, attribute, resolved=False, subgroup=subgroup)

            if result is not None:
                if resolved and type(result) is str:
                    result = self.get_resolved_value(result)
                    vtype = env_file.get_type_info(item)
                    if vtype is not None:
                        result = convert_to_type(result, vtype, item)
                return result

        for env_file in self._env_generic_files:

            result = env_file.get_value(item, attribute, resolved=False, subgroup=subgroup)

            if result is not None:
                if resolved and type(result) is str:
                    return self.get_resolved_value(result)
                return result

        # Return empty result
        return result

    def get_record_fields(self, variable, field):
        # Empty result
        result = []

        for env_file in self._env_entryid_files:
            # Wait and resolve in self rather than in env_file
            logger.debug("(get_record_field) Searching in %s",
                         env_file.__class__.__name__)
            if field == "varid":
                roots = env_file.get_nodes("entry")
            else:
                roots = env_file.get_nodes_by_id(variable)
            for root in roots:
                if root is not None:
                    if field == "raw":
                        result.append(env_file.get_raw_record(root))
                    elif field == "desc":
                        result.append(env_file.get_description(root))
                    elif field == "varid":
                        result.append(root.get("id"))
                    elif field == "group":
                        result.extend(env_file.get_groups(root))
                    elif field == "valid_values":
                        vv = env_file.get_valid_values(variable)
                        if vv:
                            result.extend(vv)
                    elif field == "file":
                        result.append(env_file.filename)

        if not result:
            for env_file in self._env_generic_files:
                roots = env_file.get_nodes(variable)
                for root in roots:
                    if root is not None:
                        if field == "raw":
                            result.append(env_file.get_raw_record(root))
                        elif field == "group":
                            result.extend(env_file.get_groups(root))
                        elif field == "file":
                            result.append(env_file.filename)

        return list(set(result))

    def get_type_info(self, item):
        result = None
        for env_file in self._env_entryid_files:
            result = env_file.get_type_info(item)
            if result is not None:
                return result

        return result

    def get_resolved_value(self, item, recurse=0):
        num_unresolved = item.count("$") if item else 0
        recurse_limit = 10
        if (num_unresolved > 0 and recurse < recurse_limit ):
            for env_file in self._env_entryid_files:
                item = env_file.get_resolved_value(item)
            if ("$" not in item):
                return item
            else:
                item = self.get_resolved_value(item, recurse=recurse+1)

        return item

    def set_value(self, item, value, subgroup=None, ignore_type=False, allow_undefined=False):
        """
        If a file has been defined, and the variable is in the file,
        then that value will be set in the file object and the file
        name is returned
        """
        if item == "CASEROOT":
            self._caseroot = value
        result = None

        for env_file in self._env_entryid_files:
            result = env_file.set_value(item, value, subgroup, ignore_type)
            if (result is not None):
                logger.debug("Will rewrite file %s %s",env_file.filename, item)
                self._env_files_that_need_rewrite.add(env_file)
                return result

        expect(allow_undefined or result is not None,
               "No variable %s found in case"%item)

    def set_valid_values(self, item, valid_values):
        """
        Update or create a valid_values entry for item and populate it
        """
        result = None
        for env_file in self._env_entryid_files:
            result = env_file.set_valid_values(item, valid_values)
            if (result is not None):
                logger.debug("Will rewrite file %s %s",env_file.filename, item)
                self._env_files_that_need_rewrite.add(env_file)
                return result

    def set_lookup_value(self, item, value):
        if item in self.lookups.keys() and self.lookups[item] is not None:
            logger.warn("Item %s already in lookups with value %s"%(item,self.lookups[item]))
        else:
            logger.debug("Setting in lookups: item %s, value %s"%(item,value))
            self.lookups[item] = value


    def _set_compset_and_pesfile(self, compset_name, files, user_compset=False, pesfile=None):
        """
        Loop through all the compset files and find the compset
        specifation file that matches either the input 'compset_name'.
        Note that the input compset name (i.e. compset_name) can be
        either a longname or an alias.  This will also set the
        compsets and pes specfication files.
        """
        science_support = []
        compset_alias = None
        components = files.get_components("COMPSETS_SPEC_FILE")
        logger.debug(" Possible components for COMPSETS_SPEC_FILE are %s" % components)

        if pesfile is not None:
            self._pesfile = pesfile

        # Loop through all of the files listed in COMPSETS_SPEC_FILE and find the file
        # that has a match for either the alias or the longname in that order
        for component in components:

            # Determine the compsets file for this component
            compsets_filename = files.get_value("COMPSETS_SPEC_FILE", {"component":component})

            # If the file exists, read it and see if there is a match for the compset alias or longname
            if (os.path.isfile(compsets_filename)):
                compsets = Compsets(compsets_filename)
                match, compset_alias, science_support = compsets.get_compset_match(name=compset_name)
                if match is not None:
                    if self._pesfile is None:
                        self._pesfile = files.get_value("PES_SPEC_FILE"     , {"component":component})
                        self.set_lookup_value("PES_SPEC_FILE"      ,
                                              files.get_value("PES_SPEC_FILE"     , {"component":component}, resolved=False))
                    self._compsetsfile = compsets_filename
                    self._compsetname = match
                    tests_filename    = files.get_value("TESTS_SPEC_FILE"   , {"component":component}, resolved=False)
                    tests_mods_dir    = files.get_value("TESTS_MODS_DIR"    , {"component":component}, resolved=False)
                    user_mods_dir     = files.get_value("USER_MODS_DIR"     , {"component":component}, resolved=False)
                    self.set_lookup_value("COMPSETS_SPEC_FILE" ,
                                   files.get_value("COMPSETS_SPEC_FILE", {"component":component}, resolved=False))
                    self._primary_component = component.upper()
                    self.set_lookup_value("TESTS_SPEC_FILE"    , tests_filename)
                    self.set_lookup_value("TESTS_MODS_DIR"     , tests_mods_dir)
                    self.set_lookup_value("USER_MODS_DIR"      , user_mods_dir)
                    logger.info("Compset longname is %s"%(match))
                    logger.info("Compset specification file is %s" %(compsets_filename))
                    logger.info("Pes     specification file is %s" %(self._pesfile))
                    return compset_alias, science_support

        if user_compset is True:
            #Do not error out for user_compset
            logger.warn("Could not find a compset match for either alias or longname in %s" %(compset_name))
            self._compsetname = compset_name
        else:
            expect(False,
                   "Could not find a compset match for either alias or longname in %s\n"%(compset_name)
                   + "You may need the --user-compset argument.")

        return None, science_support

    def _find_primary_component(self):
        """
        try to glean the primary component based on compset name
        """
        progcomps = {}
        spec = {}
        primary_component = None

        for comp in self._component_classes:

            if comp == "CPL":
                continue
            spec[comp] = self.get_value("COMP_%s"%comp)
            notprogcomps = ("D%s"%comp,"X%s"%comp,"S%s"%comp)
            if spec[comp].upper() in notprogcomps:
                progcomps[comp] = False
            else:
                progcomps[comp] = True
        expect("ATM" in progcomps and "LND" in progcomps and "OCN" in progcomps and \
               "ICE" in progcomps, " Not finding expected components in %s"%self._component_classes)
        if progcomps["ATM"] and progcomps["LND"] and progcomps["OCN"] and \
           progcomps["ICE"]:
            primary_component = "allactive"
        elif progcomps["LND"] and progcomps["OCN"] and progcomps["ICE"]:
            # this is a "J" compset
            primary_component = "allactive"
        elif progcomps["ATM"]:
            if "DOCN%SOM" in self._compsetname:
                # This is an "E" compset
                primary_component = "allactive"
            else:
                # This is an "F" or "Q" compset
                primary_component = spec["ATM"]
        elif progcomps["LND"]:
            # This is an "I" compset
            primary_component = spec["LND"]
        elif progcomps["OCN"]:
            # This is a "C" or "G" compset
            primary_component = spec["OCN"]
        elif progcomps["ICE"]:
            # This is a "D" compset
            primary_component = spec["ICE"]
        elif "GLC" in progcomps and progcomps["GLC"]:
            # This is a "TG" compset
            primary_component = spec["GLC"]
        else:
            # This is "A", "X" or "S"
            primary_component = "drv"

        return primary_component


    def get_compset_components(self):
        #If are doing a create_clone then, self._compsetname is not set yet
        components = []
        compset = self.get_value("COMPSET")
        if compset is None:
            compset = self._compsetname
        expect(compset is not None,
               "ERROR: compset is not set")
        # the first element is always the date operator - skip it
        elements = compset.split('_')[1:] # pylint: disable=maybe-no-member
        for element in elements:
            # ignore the possible BGC or TEST modifier
            if element.startswith("BGC%") or element.startswith("TEST"):
                continue
            else:
                element_component = element.split('%')[0].lower()
                if "ww" not in element_component:
                    element_component = re.sub(r'[0-9]*',"",element_component)
                components.append(element_component)
        return components


    def __iter__(self):
        for entryid_file in self._env_entryid_files:
            for key, val in entryid_file:
                if type(val) is str and '$' in val:
                    yield key, self.get_resolved_value(val)
                else:
                    yield key, val

    def set_comp_classes(self, comp_classes):
        self._component_classes = comp_classes
        for env_file in self._env_entryid_files:
            env_file.set_components(comp_classes)

    def _get_component_config_data(self, files):
        # attributes used for multi valued defaults
        # attlist is a dictionary used to determine the value element that has the most matches
        attlist = {"compset":self._compsetname, "grid":self._gridname, "cime_model":self._cime_model}

        # Determine list of component classes that this coupler/driver knows how
        # to deal with. This list follows the same order as compset longnames follow.

        # Add the group and elements for the config_files.xml
        for env_file in self._env_entryid_files:
            env_file.add_elements_by_group(files, attlist)

        drv_config_file = files.get_value("CONFIG_CPL_FILE")
        drv_comp = Component(drv_config_file)
        for env_file in self._env_entryid_files:
            env_file.add_elements_by_group(drv_comp, attributes=attlist)

        drv_config_file_model_specific = files.get_value("CONFIG_CPL_FILE_MODEL_SPECIFIC")
        drv_comp_model_specific = Component(drv_config_file_model_specific)
        for env_file in self._env_entryid_files:
            env_file.add_elements_by_group(drv_comp_model_specific, attributes=attlist)

        # loop over all elements of both component_classes and components - and get config_component_file for
        # for each component
        self.set_comp_classes(drv_comp.get_valid_model_components())

        if len(self._component_classes) > len(self._components):
            self._components.append('sesp')

        # put anything in the lookups table into existing env objects
        for key,value in self.lookups.items():
            result = self.set_value(key,value, allow_undefined=True)
            if result is not None:
                del self.lookups[key]

        for i in xrange(1,len(self._component_classes)):
            comp_class = self._component_classes[i]
            comp_name  = self._components[i-1]
            node_name = 'CONFIG_' + comp_class + '_FILE'
            # Add the group and elements for the config_files.xml
            comp_config_file = files.get_value(node_name, {"component":comp_name}, resolved=False)
            self.set_value(node_name, comp_config_file)
            comp_config_file = self.get_resolved_value(comp_config_file)
            expect(comp_config_file is not None and os.path.isfile(comp_config_file),
                   "Config file %s for component %s not found."%(comp_config_file, comp_name))
            compobj = Component(comp_config_file)
            self._component_description[comp_class] = compobj.get_description(self._compsetname)
            expect(self._component_description[comp_class] is not None,"No description found in file %s for component %s"%(comp_config_file, comp_name))
            logger.info("%s component is %s"%(comp_class, self._component_description[comp_class]))
            for env_file in self._env_entryid_files:
                env_file.add_elements_by_group(compobj, attributes=attlist)

        if self._primary_component is None:
            self._primary_component = self._find_primary_component()
            if self._pesfile is None:
                self._pesfile = files.get_value("PES_SPEC_FILE"     , {"component":self._primary_component})
            logger.info("Pes specification file is %s" %(self._pesfile))
            self.set_lookup_value("PES_SPEC_FILE", self._pesfile)

        # final cleanup of lookups table
        for key,value in self.lookups.items():
            result = self.set_value(key,value, allow_undefined=True)
            if result is not None:
                del self.lookups[key]

    def _setup_mach_pes(self, pecount, ncouplers, ninst, machine_name, mpilib):
        #--------------------------------------------
        # pe layout
        #--------------------------------------------
        mach_pes_obj = None
        # self._pesfile may already be env_mach_pes.xml if so we can just return
        gfile = GenericXML(infile=self._pesfile)
        ftype = gfile.get_id()
        expect(ftype == "env_mach_pes.xml" or ftype == "config_pes", " Do not recognize %s as a valid CIME pes file %s"%(self._pesfile, ftype))
        if ftype == "env_mach_pes.xml":
            new_mach_pes_obj = EnvMachPes(infile=self._pesfile, components=self._components)
            self.update_env(new_mach_pes_obj, "mach_pes")
            return new_mach_pes_obj.get_value("TOTALPES")
        pesobj = Pes(self._pesfile)

        match1 = re.match('(.+)x([0-9]+)', "" if pecount is None else pecount)
        match2 = re.match('([0-9]+)', "" if pecount is None else pecount)

        pes_ntasks = {}
        pes_nthrds = {}
        pes_rootpe = {}
        other      = {}

        force_tasks = None
        force_thrds = None

        if match1:
            opti_tasks = match1.group(1)
            if opti_tasks.isdigit():
                force_tasks = int(opti_tasks)
            else:
                pes_ntasks = pesobj.find_pes_layout(self._gridname, self._compsetname, machine_name,
                                                    pesize_opts=opti_tasks, mpilib=mpilib)[0]
            force_thrds = int(match1.group(2))
        elif match2:
            force_tasks = int(match2.group(1))
            pes_nthrds = pesobj.find_pes_layout(self._gridname, self._compsetname, machine_name, mpilib=mpilib)[1]
        else:
            pes_ntasks, pes_nthrds, pes_rootpe, other = pesobj.find_pes_layout(self._gridname, self._compsetname,
                                                                               machine_name, pesize_opts=pecount, mpilib=mpilib)

        if match1 or match2:
            for component_class in self._component_classes:
                if force_tasks is not None:
                    string_ = "NTASKS_" + component_class
                    pes_ntasks[string_] = force_tasks

                if force_thrds is not None:
                    string_ = "NTHRDS_" + component_class
                    pes_nthrds[string_] = force_thrds

                # Always default to zero rootpe if user forced procs and or threads
                string_ = "ROOTPE_" + component_class
                pes_rootpe[string_] = 0

        mach_pes_obj = self.get_env("mach_pes")

        if other is not None:
            for key, value in other.items():
                self.set_value(key, value)

        totaltasks = []
        for comp_class in self._component_classes:
            ntasks_str, nthrds_str, rootpe_str = "NTASKS_%s" % comp_class, "NTHRDS_%s" % comp_class, "ROOTPE_%s" % comp_class

            ntasks = pes_ntasks[ntasks_str] if ntasks_str in pes_ntasks else 1
            nthrds = pes_nthrds[nthrds_str] if nthrds_str in pes_nthrds else 1
            rootpe = pes_rootpe[rootpe_str] if rootpe_str in pes_rootpe else 0

            totaltasks.append( (ntasks + rootpe) * nthrds )

            mach_pes_obj.set_value(ntasks_str, ntasks)
            mach_pes_obj.set_value(nthrds_str, nthrds)
            mach_pes_obj.set_value(rootpe_str, rootpe)

        pesize = 1
        pes_per_node = self.get_value("PES_PER_NODE")
        for val in totaltasks:
            if val < 0:
                val = -1*val*pes_per_node
            if val > pesize:
                pesize = val
        pesize *= int(ncouplers)

        # Make sure that every component has been accounted for
        # set, nthrds and ntasks to 1 otherwise. Also set the ninst values here.
        for compclass in self._component_classes:
            key = "NINST_%s"%compclass
            if compclass == "CPL":
                mach_pes_obj.set_value(key, ncouplers)
                continue
            # ESP models are currently limited to 1 instance
            if compclass == "ESP":
                mach_pes_obj.set_value(key, 1)
            else:
                mach_pes_obj.set_value(key, ninst)

            key = "NTASKS_%s"%compclass
            if key not in pes_ntasks.keys():
                mach_pes_obj.set_value(key,1)
            key = "NTHRDS_%s"%compclass
            if compclass not in pes_nthrds.keys():
                mach_pes_obj.set_value(compclass,1)

        return pesize


    def configure(self, compset_name, grid_name, machine_name=None,
                  project=None, pecount=None, compiler=None, mpilib=None,
                  user_compset=False, pesfile=None,
                  user_grid=False, gridfile=None, ncouplers=1, ninst=1, test=False,
                  walltime=None, queue=None, output_root=None, run_unsupported=False, answer=None,
                  input_dir=None):

        #--------------------------------------------
        # compset, pesfile, and compset components
        #--------------------------------------------
        files = Files()
        compset_alias, science_support = self._set_compset_and_pesfile(compset_name, files, user_compset=user_compset, pesfile=pesfile)

        self._components = self.get_compset_components()
        #--------------------------------------------
        # grid
        #--------------------------------------------
        if user_grid is True and gridfile is not None:
            self.set_value("GRIDS_SPEC_FILE", gridfile)
        grids = Grids(gridfile)

        gridinfo = grids.get_grid_info(name=grid_name, compset=self._compsetname)

        self._gridname = gridinfo["GRID"]
        for key,value in gridinfo.items():
            logger.debug("Set grid %s %s"%(key,value))
            self.set_lookup_value(key,value)

        #--------------------------------------------
        # component config data
        #--------------------------------------------
        self._get_component_config_data(files)

        self.get_compset_var_settings()

        #--------------------------------------------
        # machine
        #--------------------------------------------
        # set machine values in env_xxx files
        machobj = Machines(machine=machine_name)
        probed_machine = machobj.probe_machine_name()
        machine_name = machobj.get_machine_name()
        self.set_value("MACH", machine_name)
        if probed_machine != machine_name and probed_machine is not None:
            logger.warning("WARNING: User-selected machine '%s' does not match probed machine '%s'" % (machine_name, probed_machine))
        else:
            logger.info("Machine is %s" % machine_name)

        nodenames = machobj.get_node_names()
        nodenames =  [x for x in nodenames if
                      '_system' not in x and '_variables' not in x and 'mpirun' not in x and\
                      'COMPILER' not in x and 'MPILIB' not in x]

        for nodename in nodenames:
            value = machobj.get_value(nodename, resolved=False)
            type_str = self.get_type_info(nodename)
            if type_str is not None:
                logger.debug("machine nodname %s value %s"%(nodename, value))
                self.set_value(nodename, convert_to_type(value, type_str, nodename))

        if compiler is None:
            compiler = machobj.get_default_compiler()
        else:
            expect(machobj.is_valid_compiler(compiler),
                   "compiler %s is not supported on machine %s" %(compiler, machine_name))

        self.set_value("COMPILER",compiler)

        if mpilib is None:
            mpilib = machobj.get_default_MPIlib({"compiler":compiler})
        else:
            expect(machobj.is_valid_MPIlib(mpilib, {"compiler":compiler}),
                   "MPIlib %s is not supported on machine %s" %(mpilib, machine_name))
        self.set_value("MPILIB",mpilib)

        machdir = machobj.get_machines_dir()
        self.set_value("MACHDIR", machdir)

        # Create env_mach_specific settings from machine info.
        env_mach_specific_obj = self.get_env("mach_specific")
        env_mach_specific_obj.populate(machobj)
        self.schedule_rewrite(env_mach_specific_obj)

        pesize = self._setup_mach_pes(pecount, ncouplers, ninst, machine_name, mpilib)

        if ncouplers > 1:
            logger.info(" Coupler has %s instances" % ncouplers)

        #--------------------------------------------
        # batch system
        #--------------------------------------------
        env_batch = self.get_env("batch")

        batch_system_type = machobj.get_value("BATCH_SYSTEM")
        batch = Batch(batch_system=batch_system_type, machine=machine_name)
        bjobs = batch.get_batch_jobs()

        env_batch.set_batch_system(batch, batch_system_type=batch_system_type)
        env_batch.create_job_groups(bjobs)
        env_batch.set_job_defaults(bjobs, pesize=pesize, walltime=walltime, force_queue=queue, allow_walltime_override=test)
        self.schedule_rewrite(env_batch)

        #--------------------------------------------
        # archiving system
        #--------------------------------------------
        env_archive = self.get_env("archive")
        infile_node = files.get_node("entry", {"id":"ARCHIVE_SPEC_FILE"})
        infile = files.get_default_value(infile_node)
        infile = self.get_resolved_value(infile)
        logger.debug("archive defaults located in %s"%infile)
        archive = Archive(infile=infile, files=files)
        archive.setup(env_archive, self._components)
        self.schedule_rewrite(env_archive)

        self.set_value("COMPSET",self._compsetname)

        self._set_pio_xml()
        logger.info(" Compset is: %s " %self._compsetname)
        logger.info(" Grid is: %s " %self._gridname )
        logger.info(" Components in compset are: %s " %self._components)

        if not test and not run_unsupported and self._cime_model == "cesm":
            if grid_name in science_support:
                logger.info("\nThis is a CESM scientifically supported compset at this resolution.\n")
            else:
                self._check_testlists(compset_alias, grid_name, files)

        # Set project id
        if project is None:
            project = get_project(machobj)
        if project is not None:
            self.set_value("PROJECT", project)
        elif machobj.get_value("PROJECT_REQUIRED"):
            expect(project is not None, "PROJECT_REQUIRED is true but no project found")

        # Resolve the CIME_OUTPUT_ROOT variable, other than this
        # we don't want to resolve variables until we need them
        if output_root is None:
            output_root = self.get_value("CIME_OUTPUT_ROOT")
        self.set_value("CIME_OUTPUT_ROOT", output_root)

        # Overwriting an existing exeroot or rundir can cause problems
        exeroot = self.get_value("EXEROOT")
        rundir = self.get_value("RUNDIR")
        for wdir in (exeroot, rundir):
            logging.debug("wdir is %s"%wdir)
            if os.path.exists(wdir):
                expect(not test, "Directory %s already exists, aborting test"% wdir)
                if answer is None:
                    response = raw_input("\nDirectory %s already exists, (r)eplace, (a)bort, or (u)se existing?"% wdir)
                else:
                    response = answer

                if response.startswith("r"):
                    shutil.rmtree(wdir)
                else:
                    expect(response.startswith("u"), "Aborting by user request")

        # miscellaneous settings
        if self.get_value("RUN_TYPE") == 'hybrid':
            self.set_value("GET_REFCASE", True)

        # Turn on short term archiving as cesm default setting
        model = get_model()
        self.set_model_version(model)
        if model == "cesm" and not test:
            self.set_value("DOUT_S",True)
            self.set_value("TIMER_LEVEL", 4)

        if test:
            self.set_value("TEST",True)

        self.initialize_derived_attributes()

        # Make sure that parallel IO is not specified if total_tasks==1
        if self.total_tasks == 1:
            for compclass in self._component_classes:
                key = "PIO_TYPENAME_%s"%compclass
                pio_typename = self.get_value(key)
                if pio_typename in ("pnetcdf", "netcdf4p"):
                    self.set_value(key, "netcdf")

        if input_dir is not None:
            self.set_value("DIN_LOC_ROOT", os.path.abspath(input_dir))

    def get_compset_var_settings(self):
        compset_obj = Compsets(infile=self.get_value("COMPSETS_SPEC_FILE"))
        matches = compset_obj.get_compset_var_settings(self._compsetname, self._gridname)
        for name, value in matches:
            if len(value) > 0:
                logger.debug("Compset specific settings: name is %s and value is %s"%(name,value))
                self.set_lookup_value(name, value)


    def set_initial_test_values(self):
        testobj = self.get_env("test")
        testobj.set_initial_values(self)

    def get_batch_jobs(self):
        batchobj = self.get_env("batch")
        return batchobj.get_jobs()

    def _set_pio_xml(self):
        pioobj = PIO()
        grid = self.get_value("GRID")
        compiler = self.get_value("COMPILER")
        mach = self.get_value("MACH")
        compset = self.get_value("COMPSET")
        mpilib = self.get_value("MPILIB")
        defaults = pioobj.get_defaults(grid=grid,compset=compset,mach=mach,compiler=compiler, mpilib=mpilib)

        for vid, value in defaults.items():
            self.set_value(vid,value)

    def _create_caseroot_tools(self):
        machines_dir = os.path.abspath(self.get_value("MACHDIR"))
        machine = self.get_value("MACH")
        toolsdir = os.path.join(self.get_value("CIMEROOT"),"scripts","Tools")
        casetools = os.path.join(self._caseroot, "Tools")
        # setup executable files in caseroot/
        exefiles = (os.path.join(toolsdir, "case.setup"),
                    os.path.join(toolsdir, "case.build"),
                    os.path.join(toolsdir, "case.submit"),
                    os.path.join(toolsdir, "case.cmpgen_namelists"),
                    os.path.join(toolsdir, "preview_namelists"),
                    os.path.join(toolsdir, "preview_run"),
                    os.path.join(toolsdir, "check_input_data"),
                    os.path.join(toolsdir, "check_case"),
                    os.path.join(toolsdir, "archive_metadata.sh"),
                    os.path.join(toolsdir, "xmlchange"),
                    os.path.join(toolsdir, "xmlquery"),
                    os.path.join(toolsdir, "pelayout"))
        try:
            for exefile in exefiles:
                destfile = os.path.join(self._caseroot,os.path.basename(exefile))
                os.symlink(exefile, destfile)
        except Exception as e:
            logger.warning("FAILED to set up exefiles: %s" % str(e))

        # set up utility files in caseroot/Tools/
        toolfiles = [os.path.join(toolsdir, "check_lockedfiles"),
                     os.path.join(toolsdir, "getTiming"),
                     os.path.join(toolsdir, "save_provenance"),
                     os.path.join(machines_dir,"Makefile"),
                     os.path.join(machines_dir,"mkSrcfiles"),
                     os.path.join(machines_dir,"mkDepends")]

        # used on Titan
        if os.path.isfile( os.path.join(toolsdir,"mdiag_reduce.csh") ):
            toolfiles.append( os.path.join(toolsdir,"mdiag_reduce.csh") )
            toolfiles.append( os.path.join(toolsdir,"mdiag_reduce.pl") )

        for toolfile in toolfiles:
            destfile = os.path.join(casetools, os.path.basename(toolfile))
            expect(os.path.isfile(toolfile)," File %s does not exist"%toolfile)
            try:
                os.symlink(toolfile, destfile)
            except Exception as e:
                logger.warning("FAILED to set up toolfiles: %s %s %s" % (str(e), toolfile, destfile))

        if get_model() == "acme":
            if os.path.exists(os.path.join(machines_dir, "syslog.%s" % machine)):
                shutil.copy(os.path.join(machines_dir, "syslog.%s" % machine), os.path.join(casetools, "mach_syslog"))
            else:
                shutil.copy(os.path.join(machines_dir, "syslog.noop"), os.path.join(casetools, "mach_syslog"))

    def _create_caseroot_sourcemods(self):
        components = self.get_compset_components()
        for component in components:
            directory = os.path.join(self._caseroot,"SourceMods","src.%s"%component)
            if not os.path.exists(directory):
                os.makedirs(directory)

        directory = os.path.join(self._caseroot, "SourceMods", "src.share")
        if not os.path.exists(directory):
            os.makedirs(directory)

        directory = os.path.join(self._caseroot,"SourceMods","src.drv")
        if not os.path.exists(directory):
            os.makedirs(directory)

        if get_model() == "cesm":
        # Note: this is CESM specific, given that we are referencing cism explitly
            if "cism" in components:
                directory = os.path.join(self._caseroot, "SourceMods", "src.cism", "glimmer-cism")
                if not os.path.exists(directory):
                    os.makedirs(directory)
                readme_file = os.path.join(directory, "README")

                str_to_write = """
                Put source mods for the glimmer-cism library in the glimmer-cism subdirectory
                This includes any files that are in the glimmer-cism subdirectory of $cimeroot/../components/cism
                Anything else (e.g., mods to source_glc or drivers) goes in this directory, NOT in glimmer-cism/"""

                with open(readme_file, "w") as fd:
                    fd.write(str_to_write)

    def create_caseroot(self, clone=False):
        if not os.path.exists(self._caseroot):
            # Make the case directory
            logger.info(" Creating Case directory %s" %self._caseroot)
            os.makedirs(self._caseroot)
        os.chdir(self._caseroot)

        # Create relevant directories in $self._caseroot
        if clone:
            newdirs = (LOCKED_DIR, "Tools")
        else:
            newdirs = ("SourceMods", LOCKED_DIR, "Buildconf", "Tools")
        for newdir in newdirs:
            os.makedirs(newdir)

        user_mods = self.get_value("%s_USER_MODS"%(self._primary_component))

        # Open a new README.case file in $self._caseroot
        append_status(" ".join(sys.argv), "README.case", caseroot=self._caseroot)
        compset_info = "Compset longname is %s"%(self.get_value("COMPSET"))
        if user_mods is not None:
            compset_info += " with user_mods directory %s"%(user_mods)
        append_status(compset_info,
                      "README.case", caseroot=self._caseroot)
        append_status("Compset specification file is %s" %
                      (self.get_value("COMPSETS_SPEC_FILE")),
                      "README.case", caseroot=self._caseroot)
        append_status("Pes     specification file is %s" %
                      (self.get_value("PES_SPEC_FILE")),
                      "README.case", caseroot=self._caseroot)
        for component_class in self._component_classes:
            if component_class == "CPL":
                append_status("Using %s coupler instances" %
                              (self.get_value("NINST_CPL")),
                              "README.case", caseroot=self._caseroot)
                continue
            comp_grid = "%s_GRID"%component_class
            append_status("Component %s is %s"%(component_class, self._component_description[component_class]),
                          "README.case", caseroot=self._caseroot)
            append_status("%s is %s"%(comp_grid,self.get_value(comp_grid)),
                          "README.case", caseroot=self._caseroot)

        if user_mods is not None:
            note = "This compset includes user_mods %s"%user_mods
            append_status(note, "README.case", caseroot=self._caseroot)
            logger.info(note)
        if not clone:
            self._create_caseroot_sourcemods()
        self._create_caseroot_tools()

    def apply_user_mods(self, user_mods_dir=None):
        """
        User mods can be specified on the create_newcase command line (usually when called from create test)
        or they can be in the compset definition, or both.
        """

        component_user_mods = self.get_value("%s_USER_MODS"%(self._primary_component))

        # This looping order will lead to the specified user_mods_dir taking
        # precedence over self._user_mods, if there are any conflicts.
        for user_mods in (component_user_mods, user_mods_dir):
            if user_mods is not None:
                if os.path.isabs(user_mods):
                    user_mods_path = user_mods
                else:
                    user_mods_path = self.get_value('USER_MODS_DIR')
                    user_mods_path = os.path.join(user_mods_path, user_mods)
                apply_user_mods(self._caseroot, user_mods_path)

    def create_clone(self, newcase, keepexe=False, mach_dir=None, project=None, cime_output_root=None):
        if cime_output_root is None:
            cime_output_root = self.get_value("CIME_OUTPUT_ROOT")
        expect(os.access(cime_output_root, os.W_OK), "Directory %s is not writable"
               "by this user.  Use the --cime-output-root flag to provide a writable "
               "scratch directory"%cime_output_root)

        newcaseroot = os.path.abspath(newcase)
        expect(not os.path.isdir(newcaseroot),
               "New caseroot directory %s already exists" % newcaseroot)
        newcasename = os.path.basename(newcaseroot)
        newcase_cimeroot = os.path.abspath(get_cime_root())

        # create clone from self to case
        clone_cimeroot = self.get_value("CIMEROOT")
        if newcase_cimeroot != clone_cimeroot:
            logger.warning(" case  CIMEROOT is %s " %newcase_cimeroot)
            logger.warning(" clone CIMEROOT is %s " %clone_cimeroot)
            logger.warning(" It is NOT recommended to clone cases from different versions of CIME.")


        # *** create case object as deepcopy of clone object ***
        srcroot = os.path.join(newcase_cimeroot,"..")
        newcase = self.copy(newcasename, newcaseroot, newsrcroot=srcroot)
        newcase.set_value("CIMEROOT", newcase_cimeroot)
        newcase.set_value("CIME_OUTPUT_ROOT", cime_output_root)

        # if we are cloning to a different user modify the output directory
        olduser = self.get_value("USER")
        newuser = os.environ.get("USER")
        if olduser != newuser:
            outputroot = self.get_value("CIME_OUTPUT_ROOT")
            outputroot = string.replace(outputroot, olduser, newuser)
            # try to make the new output directory and raise an exception
            # on any error other than directory already exists.
            try:
                os.makedirs(outputroot)
            except OSError:
                if not os.path.isdir(outputroot):
                    raise
            newcase.set_value("CIME_OUTPUT_ROOT", outputroot)
            newcase.set_value("USER", newuser)
        # determine if will use clone executable or not
        if keepexe:
            orig_exeroot = self.get_value("EXEROOT")
            newcase.set_value("EXEROOT", orig_exeroot)
            newcase.set_value("BUILD_COMPLETE","TRUE")
            orig_bld_complete = self.get_value("BUILD_COMPLETE")
            if not orig_bld_complete:
                logger.warn("\nWARNING: Creating a clone with --keepexe before building the original case may cause PIO_TYPENAME to be invalid in the clone")
                logger.warn("Avoid this message by building case one before you clone.\n")
        else:
            newcase.set_value("BUILD_COMPLETE","FALSE")

        # set machdir
        if mach_dir is not None:
            newcase.set_value("MACHDIR", mach_dir)

        # Set project id
        # Note: we do not just copy this from the clone because it seems likely that
        # users will want to change this sometimes, especially when cloning another
        # user's case. However, note that, if a project is not given, the fallback will
        # be to copy it from the clone, just like other xml variables are copied.
        if project is None:
            project = self.get_value("PROJECT", subgroup="case.run")
        if project is not None:
            newcase.set_value("PROJECT", project)

        # create caseroot
        newcase.create_caseroot(clone=True)
        newcase.flush(flushall=True)

        # copy user_ files
        cloneroot = self._caseroot
        files = glob.glob(cloneroot + '/user_*')

        for item in files:
            shutil.copy(item, newcaseroot)

        # copy SourceMod and Buildconf files
        # if symlinks exist, copy rather than follow links
        for casesub in ("SourceMods", "Buildconf"):
            shutil.copytree(os.path.join(cloneroot, casesub), os.path.join(newcaseroot, casesub)
                            , symlinks=True)

        # lock env_case.xml in new case
        lock_file("env_case.xml", newcaseroot)

        # Update README.case
        fclone   = open(cloneroot + "/README.case", "r")
        fnewcase = open(newcaseroot  + "/README.case", "a")
        fnewcase.write("\n    *** original clone README follows ****")
        fnewcase.write("\n " +  fclone.read())

        clonename = self.get_value("CASE")
        logger.info(" Successfully created new case %s from clone case %s " %(newcasename, clonename))

        case_setup(newcase)

        return newcase

    def submit_jobs(self, no_batch=False, job=None, skip_pnl=False, batch_args=None, dry_run=False):
        env_batch = self.get_env('batch')
        return env_batch.submit_jobs(self, no_batch=no_batch, job=job, skip_pnl=skip_pnl, batch_args=batch_args, dry_run=dry_run)

    def get_mpirun_cmd(self, job="case.run"):
        env_mach_specific = self.get_env('mach_specific')
        run_exe = env_mach_specific.get_value("run_exe")
        run_misc_suffix = env_mach_specific.get_value("run_misc_suffix")
        run_misc_suffix = "" if run_misc_suffix is None else run_misc_suffix
        run_suffix = run_exe + run_misc_suffix

        mpirun_cmd_override = self.get_value("MPI_RUN_COMMAND")
        if mpirun_cmd_override not in ["", None, "UNSET"]:
            return mpirun_cmd_override + " " + run_exe + " " + run_misc_suffix

        # Things that will have to be matched against mpirun element attributes
        mpi_attribs = {
            "compiler" : self.get_value("COMPILER"),
            "mpilib"   : self.get_value("MPILIB"),
            "threaded" : self.get_build_threaded(),
            "unit_testing" : False
            }

        executable, mpi_arg_list = env_mach_specific.get_mpirun(self, mpi_attribs, job=job)

        # special case for aprun
        if executable is not None and "aprun" in executable:
            aprun_args, num_nodes = get_aprun_cmd_for_case(self, run_exe)
            expect(num_nodes == self.num_nodes, "Not using optimized num nodes")
            return executable + aprun_args + " " + run_misc_suffix

        else:
            mpi_arg_string = " ".join(mpi_arg_list)

        if self.get_value("BATCH_SYSTEM") == "cobalt":
            mpi_arg_string += " : "

        return "%s %s %s" % (executable if executable is not None else "", mpi_arg_string, run_suffix)

    def set_model_version(self, model):
        version = "unknown"
        srcroot = self.get_value("SRCROOT")
        if model == "cesm":
            changelog = os.path.join(srcroot,"ChangeLog")
            if os.path.isfile(changelog):
                for line in open(changelog, "r"):
                    m = re.search("Tag name: (cesm.*)$", line)
                    if m is not None:
                        version = m.group(1)
                        break
        elif model == "acme":
            version = get_current_commit(True, srcroot)
        self.set_value("MODEL_VERSION", version)

        if version != "unknown":
            logger.info("%s model version found: %s"%(model, version))
        else:
            logger.warn("WARNING: No %s Model version found."%(model))

    def load_env(self):
        if not self._is_env_loaded:
            compiler = self.get_value("COMPILER")
            debug=self.get_value("DEBUG")
            mpilib=self.get_value("MPILIB")
            env_module = self.get_env("mach_specific")
            env_module.load_env(compiler=compiler,debug=debug, mpilib=mpilib)
            self._is_env_loaded = True

    def get_build_threaded(self):
        """
        Returns True if current settings require a threaded build/run.
        """
        force_threaded = self.get_value("BUILD_THREADED")
        return bool(force_threaded) or self.thread_count > 1

    def _check_testlists(self, compset_alias, grid_name, files):
        """
        CESM only: check the testlist file for tests of this compset grid combination
        """
        if "TESTS_SPEC_FILE" in self.lookups:
            tests_spec_file = self.get_resolved_value(self.lookups["TESTS_SPEC_FILE"])
        else:
            tests_spec_file = self.get_value("TESTS_SPEC_FILE")

        tests = Testlist(tests_spec_file, files)
        testlist = tests.get_tests(compset=compset_alias, grid=grid_name)
        testcnt = 0
        for test in testlist:
            if test["category"] == "prealpha" or test["category"] == "prebeta" or "aux_" in test["category"]:
                testcnt += 1
        if testcnt > 0:
            logger.info("\nThis compset and grid combination is not scientifically supported, however it is used in %d tests.\n"%(testcnt))
        else:
            expect(False, "\nThis compset and grid combination is untested in CESM.  "
                   "Override this warning with the --run-unsupported option to create_newcase.",
                   error_prefix="STOP: ")

    def set_file(self, xmlfile):
        """
        force the case object to consider only xmlfile
        """
        expect(os.path.isfile(xmlfile), "Could not find file %s"%xmlfile)

        gfile = GenericXML(infile=xmlfile)
        ftype = gfile.get_id()
        components = self.get_value("COMP_CLASSES")
        logger.warn("setting case file to %s"%xmlfile)
        new_env_file = None
        for env_file in self._env_entryid_files:
            if os.path.basename(env_file.filename) == ftype:
                if ftype == "env_run.xml":
                    new_env_file = EnvRun(infile=xmlfile, components=components)
                elif ftype == "env_build.xml":
                    new_env_file = EnvBuild(infile=xmlfile, components=components)
                elif ftype == "env_case.xml":
                    new_env_file = EnvCase(infile=xmlfile, components=components)
                elif ftype == "env_mach_pes.xml":
                    new_env_file = EnvMachPes(infile=xmlfile, components=components)
                elif ftype == "env_batch.xml":
                    new_env_file = EnvBatch(infile=xmlfile)
                elif ftype == "env_test.xml":
                    new_env_file = EnvTest(infile=xmlfile)
            if new_env_file is not None:
                self._env_entryid_files = []
                self._env_generic_files = []
                self._env_entryid_files.append(new_env_file)
                break
        if new_env_file is None:
            for env_file in self._env_generic_files:
                if os.path.basename(env_file.filename) == ftype:
                    if ftype == "env_archive.xml":
                        new_env_file = EnvArchive(infile=xmlfile)
                    elif ftype == "env_mach_specific.xml":
                        new_env_file = EnvMachSpecific(infile=xmlfile)
                    else:
                        expect(False, "No match found for file type %s"%ftype)
                if new_env_file is not None:
                    self._env_entryid_files = []
                    self._env_generic_files = []
                    self._env_generic_files.append(new_env_file)
                    break

        self._files = self._env_entryid_files + self._env_generic_files

    def update_env(self, new_object, env_file):
        """
        Replace a case env object file
        """
        old_object = self.get_env(env_file)
        new_object.filename = old_object.filename
        if old_object in self._env_entryid_files:
            self._env_entryid_files.remove(old_object)
            self._env_entryid_files.append(new_object)
        elif old_object in self._env_generic_files:
            self._env_generic_files.remove(old_object)
            self._env_generic_files.append(new_object)
        if old_object in self._env_files_that_need_rewrite:
            self._env_files_that_need_rewrite.remove(old_object)
        self._files.remove(old_object)
        self._files.append(new_object)
        self.schedule_rewrite(new_object)
