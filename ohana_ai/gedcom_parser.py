"""
GEDCOM parser for extracting genealogical data from GEDCOM files.
Supports standard GEDCOM format with date parsing and relationship extraction.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import re
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class Individual:
    """Represents an individual in the family tree."""
    id: str
    given_names: str = ""
    surname: str = ""
    gender: str = "U"  # M, F, or U (unknown)
    birth_date: Optional[str] = None
    birth_year: Optional[int] = None
    birth_place: str = ""
    death_date: Optional[str] = None
    death_year: Optional[int] = None
    death_place: str = ""
    occupation: str = ""
    note: str = ""
    
    # Relationships
    parent_families: Set[str] = field(default_factory=set)
    spouse_families: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        """Extract years from birth and death dates."""
        if self.birth_date:
            self.birth_year = self._extract_year(self.birth_date)
        if self.death_date:
            self.death_year = self._extract_year(self.death_date)
    
    @staticmethod
    def _extract_year(date_str: str) -> Optional[int]:
        """Extract year from GEDCOM date string."""
        if not date_str:
            return None
        
        # Common GEDCOM date patterns
        patterns = [
            r'\b(\d{4})\b',  # 4-digit year
            r'\b(\d{1,2})\s+\w+\s+(\d{4})\b',  # DD MMM YYYY
            r'\b\w+\s+(\d{4})\b',  # MMM YYYY
        ]
        
        for pattern in patterns:
            match = re.search(pattern, date_str)
            if match:
                year = match.group(-1)  # Last group is the year
                try:
                    return int(year)
                except ValueError:
                    continue
        
        return None
    
    @property
    def full_name(self) -> str:
        """Get full name combining given names and surname."""
        parts = []
        if self.given_names:
            parts.append(self.given_names)
        if self.surname:
            parts.append(self.surname)
        return " ".join(parts)
    
    @property
    def is_alive_at_year(self) -> callable:
        """Return function to check if person was alive at given year."""
        def check_alive(year: int) -> bool:
            if self.birth_year and year < self.birth_year:
                return False
            if self.death_year and year > self.death_year:
                return False
            return True
        return check_alive

@dataclass
class Family:
    """Represents a family unit with parents and children."""
    id: str
    husband_id: Optional[str] = None
    wife_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    marriage_date: Optional[str] = None
    marriage_year: Optional[int] = None
    marriage_place: str = ""
    divorce_date: Optional[str] = None
    divorce_year: Optional[int] = None
    
    def __post_init__(self):
        """Extract years from marriage and divorce dates."""
        if self.marriage_date:
            self.marriage_year = Individual._extract_year(self.marriage_date)
        if self.divorce_date:
            self.divorce_year = Individual._extract_year(self.divorce_date)
    
    @property
    def parent_ids(self) -> List[str]:
        """Get list of parent IDs."""
        parents = []
        if self.husband_id:
            parents.append(self.husband_id)
        if self.wife_id:
            parents.append(self.wife_id)
        return parents

class GEDCOMParser:
    """Parser for GEDCOM genealogy files."""
    
    def __init__(self):
        self.individuals: Dict[str, Individual] = {}
        self.families: Dict[str, Family] = {}
        self.current_record = None
        self.current_id = None
        
    def parse_file(self, filepath: str) -> Tuple[Dict[str, Individual], Dict[str, Family]]:
        """Parse a GEDCOM file and return individuals and families."""
        logger.info(f"Parsing GEDCOM file: {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
                lines = file.readlines()
        except UnicodeDecodeError:
            # Try different encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    with open(filepath, 'r', encoding=encoding) as file:
                        lines = file.readlines()
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError(f"Could not decode file {filepath}")
        
        self._parse_lines(lines)
        self._link_relationships()
        
        logger.info(f"Parsed {len(self.individuals)} individuals and {len(self.families)} families")
        return self.individuals, self.families
    
    def _parse_lines(self, lines: List[str]) -> None:
        """Parse GEDCOM lines and extract records."""
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split(' ', 2)
            if len(parts) < 2:
                continue
                
            level = int(parts[0])
            tag = parts[1]
            value = parts[2] if len(parts) > 2 else ""
            
            self._process_line(level, tag, value)
    
    def _process_line(self, level: int, tag: str, value: str) -> None:
        """Process a single GEDCOM line."""
        if level == 0:
            if tag.startswith('@') and tag.endswith('@'):
                self.current_id = tag[1:-1]  # Remove @ symbols
                if value == "INDI":
                    self.current_record = Individual(id=self.current_id)
                    self.individuals[self.current_id] = self.current_record
                elif value == "FAM":
                    self.current_record = Family(id=self.current_id)
                    self.families[self.current_id] = self.current_record
                else:
                    self.current_record = None
            else:
                self.current_record = None
        
        elif level == 1 and self.current_record:
            self._process_level1_tag(tag, value)
        
        elif level == 2 and self.current_record:
            self._process_level2_tag(tag, value)
    
    def _process_level1_tag(self, tag: str, value: str) -> None:
        """Process level 1 tags."""
        if isinstance(self.current_record, Individual):
            if tag == "NAME":
                self._parse_name(value)
            elif tag == "SEX":
                self.current_record.gender = value
            elif tag in ["BIRT", "DEAT"]:
                self._current_event = tag
            elif tag == "OCCU":
                self.current_record.occupation = value
            elif tag == "NOTE":
                self.current_record.note = value
            elif tag == "FAMC":
                family_id = value.strip('@')
                self.current_record.parent_families.add(family_id)
            elif tag == "FAMS":
                family_id = value.strip('@')
                self.current_record.spouse_families.add(family_id)
        
        elif isinstance(self.current_record, Family):
            if tag == "HUSB":
                self.current_record.husband_id = value.strip('@')
            elif tag == "WIFE":
                self.current_record.wife_id = value.strip('@')
            elif tag == "CHIL":
                self.current_record.children_ids.append(value.strip('@'))
            elif tag in ["MARR", "DIV"]:
                self._current_event = tag
    
    def _process_level2_tag(self, tag: str, value: str) -> None:
        """Process level 2 tags."""
        if not hasattr(self, '_current_event'):
            return
            
        event = self._current_event
        
        if isinstance(self.current_record, Individual):
            if event == "BIRT":
                if tag == "DATE":
                    self.current_record.birth_date = value
                elif tag == "PLAC":
                    self.current_record.birth_place = value
            elif event == "DEAT":
                if tag == "DATE":
                    self.current_record.death_date = value
                elif tag == "PLAC":
                    self.current_record.death_place = value
        
        elif isinstance(self.current_record, Family):
            if event == "MARR":
                if tag == "DATE":
                    self.current_record.marriage_date = value
                elif tag == "PLAC":
                    self.current_record.marriage_place = value
            elif event == "DIV":
                if tag == "DATE":
                    self.current_record.divorce_date = value
    
    def _parse_name(self, name_str: str) -> None:
        """Parse GEDCOM name format: Given names /Surname/."""
        if not isinstance(self.current_record, Individual):
            return
            
        # GEDCOM format: Given names /Surname/
        match = re.match(r'^([^/]*?)\s*/([^/]*?)/$', name_str.strip())
        if match:
            self.current_record.given_names = match.group(1).strip()
            self.current_record.surname = match.group(2).strip()
        else:
            # Fallback: treat entire string as given names
            self.current_record.given_names = name_str.strip()
    
    def _link_relationships(self) -> None:
        """Link family relationships after parsing."""
        for family in self.families.values():
            # Link parents to children
            for child_id in family.children_ids:
                if child_id in self.individuals:
                    child = self.individuals[child_id]
                    child.parent_families.add(family.id)
            
            # Link spouses to family
            if family.husband_id and family.husband_id in self.individuals:
                husband = self.individuals[family.husband_id]
                husband.spouse_families.add(family.id)
            
            if family.wife_id and family.wife_id in self.individuals:
                wife = self.individuals[family.wife_id]
                wife.spouse_families.add(family.id)

def parse_gedcom_file(filepath: str) -> Tuple[Dict[str, Individual], Dict[str, Family]]:
    """Convenience function to parse a GEDCOM file."""
    parser = GEDCOMParser()
    return parser.parse_file(filepath)