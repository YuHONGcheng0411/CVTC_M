#!/usr/bin/perl
use strict;
use warnings;

# Check for correct number of command-line arguments
die "Usage: perl $0 list.txt BIOGRID_INTACT.HUMAN.txt > list.ppi.txt\n" unless @ARGV == 2;

# Read gene list into a hash for quick lookup
my %genes;
open(my $in, '<', $ARGV[0]) or die "Cannot open $ARGV[0]: $!\n";
while (<$in>) {
    chomp;
    # Match gene ID at the start of the line
    if (/^(\S+)/) {
        my $gene = uc($1);  # Convert to uppercase for consistency
        next if $gene eq "NULL";  # Skip NULL entries
        $genes{$gene} = 1;
    }
}
close($in);

# Process PPI data and output interactions
my %interactions;
open(my $ppi, '<', $ARGV[1]) or die "Cannot open $ARGV[1]: $!\n";
while (<$ppi>) {
    chomp;
    # Match two gene IDs in tab-separated format
    if (/^(\S+)\s+(\S+)/) {
        my ($gene1, $gene2) = (uc($1), uc($2));  # Convert to uppercase
        next if $gene1 eq $gene2;  # Skip self-interactions
        # Check if both genes exist in the gene list and interaction is not already recorded
        next unless exists $genes{$gene1} && exists $genes{$gene2};
        next if exists $interactions{$gene1}{$gene2};
        # Record interaction in both directions to ensure symmetry
        $interactions{$gene1}{$gene2} = 1;
        $interactions{$gene2}{$gene1} = 1;
        print "$gene1\t$gene2\n";
    }
}
close($ppi);