import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    # Create list to keep track of all the different probabilities to be multiplied together
    probabilities = []
    # Creates dictionary to map each person to their number of genes we want to compute the probability for
    genes = dict()

    """Creates 2D lists containing the probability of a child having one, two or none genes 
       The rows in each list represent # of mother's genes, cell within each row represents # of father's genes
       For example, one_gene_probabilities[0][2] represents probability of child having 1 gene
       when mother has 0 genes and father has 2 genes. 
    """
    one_gene_probabilities = [[0.0198, 0.5, 0.9802], [0.5, 0.5, 0.5], [0.9802, 0.5, 0.0198]]
    two_gene_probabilities = [[0.0001, 0.005, 0.0099], [0.005, 0.25, 0.495], [0.0099, 0.495, 0.9801]]
    zero_gene_probabilities = [[0.9801, 0.495, 0.0099], [0.495, 0.25, 0.005], [0.0099, 0.005, 0.0001]]

    # Access every person in people dictionary
    for person in people:
        # Check if person is in one_gene. if so, adds probability of person having 1 gene to list.
        if person in one_gene:
            genes[person] = 1
            # If the person does not have parent data, adds unconditional probability of one gene
            if people[person]["mother"] is None:
                probabilities.append(0.03)
            # Otherwise, adds conditional probability based on parent data
            else:
                # Finds whether parents are in one_gene, two_genes, or neither
                mother_genes = find_parent_genes(people, person, "mother", one_gene, two_genes)
                father_genes = find_parent_genes(people, person, "father", one_gene, two_genes)

                # Adds the probability that person has one gene given parent's numbers of genes
                probabilities.append(one_gene_probabilities[mother_genes][father_genes])

        # Check if person is in two_genes. if so, adds probability of person having 2 genes to list.
        elif person in two_genes:
            genes[person] = 2
            # If the person does not have parent data, adds unconditional probability of one gene
            if people[person]["mother"] is None:
                probabilities.append(0.01)
            # Otherwise, adds conditional probability based on parent data
            else:
                # Finds whether parents are in one_gene, two_genes, or neither
                mother_genes = find_parent_genes(people, person, "mother", one_gene, two_genes)
                father_genes = find_parent_genes(people, person, "father", one_gene, two_genes)

                # Adds the probability that person has two genes given parent's numbers of genes
                probabilities.append(two_gene_probabilities[mother_genes][father_genes])

        # If person is not in one_gene or two_genes, adds probability of person have 0 genes to list.
        else:
            genes[person] = 0
            # If the person does not have parent data, adds unconditional probability of one gene
            if people[person]["mother"] is None:
                probabilities.append(0.96)
            # Otherwise, adds conditional probability based on parent data
            else:
                # Finds whether parents are in one_gene, two_genes, or neither
                mother_genes = find_parent_genes(people, person, "mother", one_gene, two_genes)
                father_genes = find_parent_genes(people, person, "father", one_gene, two_genes)

                # Adds the probability that person has zero genes given parent's numbers of genes
                probabilities.append(zero_gene_probabilities[mother_genes][father_genes])

        # If person is in have_trait
        if person in have_trait:
            num_genes = genes[person]
            # Adds probability that they have the trait with given number of genes
            probabilities.append(PROBS["trait"][num_genes][True])
        else:
            num_genes = genes[person]
            # Adds probability that they don't have the trait with given number of genes
            probabilities.append(PROBS["trait"][num_genes][False])

    joint_prob = 1
    # Traverses through list "probabilities" and multiplies all values together to get joint probability
    for prob in probabilities:
        joint_prob *= prob
    return joint_prob


def find_parent_genes(people, person, parent, one_gene, two_genes):
    """ Returns the number of genes the parent, ("mother" or "father"),
        of given person has"""
    if people[person][parent] in one_gene:
        return 1
    elif people[person][parent] in two_genes:
        return 2
    else:
        return 0


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    # Access each person in probabilities
    for person in probabilities:
        # If the person is in one gene, updates corresponding probability
        if person in one_gene:
            probabilities[person]["gene"][1] += p
        # If the person is in two_genes, updates corresponding probability
        elif person in two_genes:
            probabilities[person]["gene"][2] += p
        # If the person is in neither one_gene or two_genes, updates corresponding probability
        else:
            probabilities[person]["gene"][0] += p
        # If the person is in have_trait, updates corresponding probability
        if person in have_trait:
            probabilities[person]["trait"][True] += p
        # If the person is not in have_trait, updates corresponding probability
        else:
            probabilities[person]["trait"][False] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """

    # Access each person in probabilities
    for person in probabilities:
        # stores sum of all probabilities in "gene" dictionary in total
        total = probabilities[person]["gene"][0] + probabilities[person]["gene"][1] + probabilities[person]["gene"][2]
        # divides each probability by the total to get percentage values, which will all add up to 1
        probabilities[person]["gene"][0] /= total
        probabilities[person]["gene"][1] /= total
        probabilities[person]["gene"][2] /= total

        # stores sum of all probabilities in "trait" dictionary in total
        total = probabilities[person]["trait"][True] + probabilities[person]["trait"][False]
        # divides each probability by the total to get percentage values, which will all add up to 1
        probabilities[person]["trait"][True] /= total
        probabilities[person]["trait"][False] /= total



if __name__ == "__main__":
    main()
