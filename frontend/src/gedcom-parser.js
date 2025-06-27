export const parseGedcom = (gedcomContent) => {
    const individuals = {};
    const families = {};
    let currentIndividual = null;
    let currentFamily = null;

    const lines = gedcomContent.split(/\r?\n/);

    lines.forEach(line => {
        const match = line.match(/^(\d+)\s+(@[^@]+@)?\s*(\w+)\s*(.*)$/);
        if (!match) return;

        const level = parseInt(match[1]);
        const xref = match[2];
        const tag = match[3];
        const value = match[4].trim();

        if (level === 0) {
            if (tag === 'INDI') {
                currentIndividual = { id: xref, name: '', sex: '', famc: [], fams: [] };
                individuals[xref] = currentIndividual;
                currentFamily = null;
            } else if (tag === 'FAM') {
                currentFamily = { id: xref, husb: '', wife: '', chil: [] };
                families[xref] = currentFamily;
                currentIndividual = null;
            } else {
                currentIndividual = null;
                currentFamily = null;
            }
        } else if (level === 1) {
            if (currentIndividual) {
                if (tag === 'NAME') {
                    currentIndividual.name = value;
                } else if (tag === 'SEX') {
                    currentIndividual.sex = value;
                } else if (tag === 'FAMC') {
                    currentIndividual.famc.push(value);
                } else if (tag === 'FAMS') {
                    currentIndividual.fams.push(value);
                }
            } else if (currentFamily) {
                if (tag === 'HUSB') {
                    currentFamily.husb = value;
                } else if (tag === 'WIFE') {
                    currentFamily.wife = value;
                } else if (tag === 'CHIL') {
                    currentFamily.chil.push(value);
                }
            }
        }
    });

    return { individuals, families };
};