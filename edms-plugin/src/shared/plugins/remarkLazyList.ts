// plugins/remarkLazyList.ts
import {visit} from 'unist-util-visit'
import type {Plugin} from 'unified'
import type {Root, Paragraph, Text} from 'mdast'

export const remarkLazyList: Plugin<[], Root> = () => {
    return (tree: Root) => {
        let currentList: any = null
        let currentItems: string[] = []

        visit(tree, 'paragraph', (node: Paragraph, index: number, parent: any) => {
            const textNode = node.children[0] as Text
            if (!textNode?.value) return

            const lines = textNode.value.split('\n')
            const isLazyList = lines.some(line =>
                /^\s*\d+[.)]\s/.test(line) ||
                /^\s*[0-9]️?\u20E3?\s/.test(line) ||
                /^\s*[-*•]\s/.test(line)
            )

            if (isLazyList && lines.length > 1) {
                const isOrdered = /^\s*\d+[.)]/.test(lines[0]) || /^\s*[0-9]️?\u20E3?/.test(lines[0])
                const listItems = lines
                    .filter(line => line.trim())
                    .map(line => ({
                        type: 'listItem',
                        children: [{
                            type: 'paragraph',
                            children: [{
                                type: 'text',
                                value: line.replace(/^\s*\d+[.)]\s*/, '')
                                    .replace(/^\s*[0-9]️?\u20E3?\s*/, '')
                                    .replace(/^\s*[-*•]\s*/, '')
                            }]
                        }]
                    }))

                const listNode = {
                    type: 'list',
                    ordered: isOrdered,
                    children: listItems
                }

                parent.children.splice(index, 1, listNode)
                currentList = null
                currentItems = []
            }
        })
    }
}