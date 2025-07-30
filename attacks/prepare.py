import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def debug_data_split():
    """Debug the data split to find the source of duplicates"""

    print("=== DEBUGGING DATA SPLIT ===")

    # Load the problematic data
    X_member = np.load("attack_data/X_member.npy")
    X_nonmember = np.load("attack_data/X_nonmember.npy")

    print(f"Member data shape: {X_member.shape}")
    print(f"Non-member data shape: {X_nonmember.shape}")

    # Find exact duplicates
    member_strings = [str(row) for row in X_member]
    nonmember_strings = [str(row) for row in X_nonmember]

    member_set = set(member_strings)
    nonmember_set = set(nonmember_strings)

    print(f"Unique member samples: {len(member_set)}")
    print(f"Unique non-member samples: {len(nonmember_set)}")

    # Find overlaps
    overlap = member_set.intersection(nonmember_set)
    print(f"Overlapping samples: {len(overlap)}")

    if overlap:
        print("\n=== ANALYZING OVERLAPS ===")
        # Find indices of overlapping samples
        overlap_indices_member = []
        overlap_indices_nonmember = []

        for i, sample_str in enumerate(member_strings):
            if sample_str in overlap:
                overlap_indices_member.append(i)

        for i, sample_str in enumerate(nonmember_strings):
            if sample_str in overlap:
                overlap_indices_nonmember.append(i)

        print(f"Overlap indices in member data: {overlap_indices_member[:10]}...")  # Show first 10
        print(f"Overlap indices in nonmember data: {overlap_indices_nonmember[:10]}...")

        # Show a few examples
        print(f"\nFirst overlapping sample:")
        print(f"Member[{overlap_indices_member[0]}]: {X_member[overlap_indices_member[0]]}")
        print(f"NonMember[{overlap_indices_nonmember[0]}]: {X_nonmember[overlap_indices_nonmember[0]]}")

        # Check if duplicates are clustered (might indicate systematic error)
        print(f"\nAre duplicates clustered?")
        print(f"Member overlap indices spread: {min(overlap_indices_member)} to {max(overlap_indices_member)}")
        print(f"NonMember overlap indices spread: {min(overlap_indices_nonmember)} to {max(overlap_indices_nonmember)}")

    # Check for internal duplicates
    print(f"\n=== CHECKING INTERNAL DUPLICATES ===")
    member_internal_dups = len(member_strings) - len(member_set)
    nonmember_internal_dups = len(nonmember_strings) - len(nonmember_set)

    print(f"Internal duplicates in member data: {member_internal_dups}")
    print(f"Internal duplicates in nonmember data: {nonmember_internal_dups}")

    # Basic statistics
    print(f"\n=== BASIC STATISTICS ===")
    print(f"Member data:")
    print(f"  Mean: {X_member.mean(axis=0)[:5]}...")  # Show first 5 features
    print(f"  Std: {X_member.std(axis=0)[:5]}...")
    print(f"  Min: {X_member.min()}")
    print(f"  Max: {X_member.max()}")

    print(f"Non-member data:")
    print(f"  Mean: {X_nonmember.mean(axis=0)[:5]}...")
    print(f"  Std: {X_nonmember.std(axis=0)[:5]}...")
    print(f"  Min: {X_nonmember.min()}")
    print(f"  Max: {X_nonmember.max()}")


def create_proper_split():
    """Create a proper train/test split for MIA"""

    print("\n=== CREATING PROPER SPLIT ===")

    # You need to replace this with your actual data loading
    # This is just an example of how to do it properly

    print("IMPORTANT: You need to modify this function to load your original data!")
    print("Replace the following lines with your actual data loading code:")
    print("# X, y = load_your_original_data()")
    print("# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)")

    # Example (you need to replace this):
    """
    # Load your original dataset
    X, y = load_your_original_data()  # Replace with your actual data loading

    # Proper split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.5,  # 50/50 split
        random_state=42,  # For reproducibility
        stratify=y  # Ensure balanced classes in both sets
    )

    # Save for MIA
    np.save("attack_data/X_member.npy", X_train)
    np.save("attack_data/X_nonmember.npy", X_test)
    np.save("attack_data/y_member.npy", y_train)
    np.save("attack_data/y_nonmember.npy", y_test)

    print("Proper split created and saved!")

    # Verify no overlaps
    member_set = set(str(row) for row in X_train)
    nonmember_set = set(str(row) for row in X_test)
    overlap = member_set.intersection(nonmember_set)

    print(f"New split verification:")
    print(f"  Member samples: {len(X_train)}")
    print(f"  Non-member samples: {len(X_test)}")
    print(f"  Overlaps: {len(overlap)}")

    if len(overlap) == 0:
        print("✅ Perfect! No overlaps found.")
    else:
        print("❌ Still have overlaps - check your data!")
    """


def remove_duplicates_quick_fix():
    """Quick fix: remove duplicates from existing data"""

    print("\n=== QUICK FIX: REMOVING DUPLICATES ===")

    X_member = np.load("attack_data/X_member.npy")
    X_nonmember = np.load("attack_data/X_nonmember.npy")

    # Find duplicates
    member_strings = [str(row) for row in X_member]
    nonmember_strings = [str(row) for row in X_nonmember]

    member_set = set(member_strings)
    nonmember_set = set(nonmember_strings)
    overlap = member_set.intersection(nonmember_set)

    # Remove duplicates from non-member data
    clean_nonmember_indices = []
    for i, sample_str in enumerate(nonmember_strings):
        if sample_str not in overlap:
            clean_nonmember_indices.append(i)

    X_nonmember_clean = X_nonmember[clean_nonmember_indices]

    print(f"Original non-member size: {len(X_nonmember)}")
    print(f"Clean non-member size: {len(X_nonmember_clean)}")
    print(f"Removed samples: {len(X_nonmember) - len(X_nonmember_clean)}")

    # Verify
    member_set_clean = set(str(row) for row in X_member)
    nonmember_set_clean = set(str(row) for row in X_nonmember_clean)
    overlap_clean = member_set_clean.intersection(nonmember_set_clean)

    print(f"Remaining overlaps: {len(overlap_clean)}")

    if len(overlap_clean) == 0:
        # Save cleaned data
        np.save("attack_data/X_nonmember_clean.npy", X_nonmember_clean)
        print("✅ Cleaned data saved to attack_data/X_nonmember_clean.npy")
        print("Update your attack script to use X_nonmember_clean.npy instead!")
    else:
        print("❌ Still have overlaps after cleaning")


if __name__ == "__main__":
    debug_data_split()
    remove_duplicates_quick_fix()
    create_proper_split()